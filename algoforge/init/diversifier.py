"""Init phase: diverse seed generation and archive initialization."""

import asyncio
import json
import logging
import random
import types
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import AlgoforgeConfig
from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..behavior import BehaviorExtractor
from ..llm import PromptBuilder, ProgramWithScore, OutputMode, get_llm_client
from ..utils import ResilientProcessPool, extract_code, extract_fn_name

logger = logging.getLogger(__name__)

DIVERSITY_SEED_PROMPT = """# {problem_title}

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Your Task: ALGORITHMIC DIVERSITY

You MUST design a solution using a **FUNDAMENTALLY DIFFERENT ALGORITHM** than the existing seeds.

**DO NOT:**
- Make minor variations or parameter tweaks to existing approaches
- Use the same core algorithm with different constants
- Reorder or refactor existing logic

**DO:**
- Analyze what algorithmic paradigm each existing seed uses
- Identify what aspects of the problem they exploit (or ignore)
- Design from first principles using a completely different strategy
- Think about what information in the problem they are NOT using
- Consider entirely different ways to model or decompose the problem

The goal is to explore different regions of the algorithm design space. A population of diverse algorithms will outperform a population of similar ones.

## Existing Seeds (analyze their algorithms, then do something DIFFERENT):
{existing_seeds}

## Output
Output ONLY the complete Python code in a ```python block.
"""


def _evaluate_code(code: str, score_fn, inputs: list, fn_name: str) -> dict:
    """Runs in subprocess: parse code, extract callable, call score_fn."""
    namespace = {}
    try:
        exec(code, namespace)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}
    except MemoryError:
        return {"error": "MemoryError during code execution"}

    fn = namespace.get(fn_name)
    if not isinstance(fn, types.FunctionType):
        return {"error": f"Function '{fn_name}' not found (got {type(fn).__name__})"}

    try:
        return score_fn(fn, inputs)
    except MemoryError:
        return {"error": "MemoryError during scoring"}


@dataclass
class PredefinedCentroidsResult:
    """Result from loading predefined centroids."""
    centroids: np.ndarray
    raw_feature_data: dict[str, list[float]]
    bounds: Optional[dict[str, tuple[float, float]]] = None


def _load_predefined_centroids(json_path: str, feature_names: list[str]) -> PredefinedCentroidsResult:
    """Load predefined centroids from JSON.

    Supports two formats:
    1. New format with explicit bounds:
       {"bounds": {"feature": [min, max], ...}, "centroids": [...]}
    2. Legacy format (list of centroid objects):
       [{"name": ..., "vector": {...}}, ...]

    When bounds are provided, uses deterministic min-max normalization.
    Otherwise falls back to z-score normalization based on centroid data.

    Returns PredefinedCentroidsResult with centroids_array, raw_feature_data,
    and optional bounds dict.
    """
    with open(json_path) as f:
        data = json.load(f)

    # Map JSON feature names to extractor feature names
    feature_map = {
        "max_loop_nesting": "loop_nesting_max",
        "loop_nesting_max": "loop_nesting_max",
        "math_operators": "math_operators",
        "cyclomatic_complexity": "cyclomatic_complexity",
        "ast_depth": "ast_depth",
    }

    # Check for new format with explicit bounds
    bounds: Optional[dict[str, tuple[float, float]]] = None
    centroid_entries: list[dict]

    if isinstance(data, dict) and "bounds" in data and "centroids" in data:
        # New format: {"bounds": {...}, "centroids": [...]}
        raw_bounds = data["bounds"]
        bounds = {}
        for json_feat, (lo, hi) in raw_bounds.items():
            mapped_feat = feature_map.get(json_feat, json_feat)
            if mapped_feat in feature_names:
                bounds[mapped_feat] = (float(lo), float(hi))
        centroid_entries = data["centroids"]
    else:
        # Legacy format: list of centroid objects
        centroid_entries = data

    # Collect raw values per mapped feature
    raw_feature_data: dict[str, list[float]] = {f: [] for f in feature_names}
    for entry in centroid_entries:
        vec = entry["vector"]
        for json_feat, val in vec.items():
            mapped_feat = feature_map.get(json_feat, json_feat)
            if mapped_feat in raw_feature_data:
                raw_feature_data[mapped_feat].append(val)

    n_dims = len(feature_names)
    centroids = []

    if bounds is not None:
        # Deterministic mode: use provided bounds for min-max normalization
        for entry in centroid_entries:
            vec = entry["vector"]
            normalized = np.full(n_dims, 0.5)

            for json_feat, val in vec.items():
                mapped_feat = feature_map.get(json_feat, json_feat)
                if mapped_feat in feature_names and mapped_feat in bounds:
                    idx = feature_names.index(mapped_feat)
                    lo, hi = bounds[mapped_feat]
                    normalized[idx] = np.clip((val - lo) / (hi - lo), 0.0, 1.0)

            centroids.append(normalized)
    else:
        # Legacy mode: z-score normalization
        def zscore_to_01(z: float) -> float:
            z = max(-10, min(10, z))
            return 1.0 / (1.0 + np.exp(-z))

        feature_stats = {}
        for feat, vals in raw_feature_data.items():
            if vals:
                mean = np.mean(vals)
                std = max(np.std(vals, ddof=1), 0.1)
                feature_stats[feat] = (mean, std)
            else:
                feature_stats[feat] = (0.0, 1.0)

        for entry in centroid_entries:
            vec = entry["vector"]
            normalized = np.full(n_dims, 0.5)

            for json_feat, val in vec.items():
                mapped_feat = feature_map.get(json_feat, json_feat)
                if mapped_feat in feature_names:
                    idx = feature_names.index(mapped_feat)
                    mean, std = feature_stats[mapped_feat]
                    z = (val - mean) / std
                    normalized[idx] = zscore_to_01(z)

            centroids.append(normalized)

    return PredefinedCentroidsResult(
        centroids=np.array(centroids),
        raw_feature_data=raw_feature_data,
        bounds=bounds,
    )


class Diversifier:
    """Handles init phase: diverse seed generation and archive population."""

    def __init__(
        self,
        config: AlgoforgeConfig,
        executor: ResilientProcessPool,
    ):
        self.config = config
        self.executor = executor
        self.total_cost = 0.0
        self.best_score = 0.0

    async def _cascade_eval(self, code: str, fn_name: str) -> dict:
        """Evaluate with cascade: quick test first, full eval only if promising."""
        cascade = self.config.cascade
        if cascade.enabled and cascade.quick_inputs:
            quick_result = await self.executor.run(
                _evaluate_code,
                code,
                self.config.score_fn,
                cascade.quick_inputs,
                fn_name,
                timeout=cascade.quick_timeout
            )
            if "error" in quick_result:
                return quick_result
            quick_score = quick_result.get('score', 0)
            threshold = self.best_score * cascade.min_score_ratio
            if quick_score < threshold:
                return {"error": f"Cascade rejected: {quick_score:.1f} < {threshold:.1f}"}

        result = await self.executor.run(
            _evaluate_code,
            code,
            self.config.score_fn,
            self.config.inputs,
            fn_name,
            timeout=self.config.pipeline.eval_timeout
        )
        if "error" not in result:
            score = result.get('score', 0)
            if score > self.best_score:
                self.best_score = score
        return result

    async def run(
        self,
        pool: CVTMAPElitesPool,
        seed_program: str,
        seed_result: dict,
        extractor: BehaviorExtractor,
    ) -> float:
        """
        Run init phase: generate diverse seeds, expand with variants, build centroids.

        Returns total cost incurred during init.
        """
        fn_name = extract_fn_name(self.config.function_signature)
        seed_score = seed_result.get('score', 0.0)
        self.best_score = seed_score

        # Phase 1: Generate diverse seeds
        diverse_seeds = await self._generate_diverse_seeds(
            seed_program, seed_score, seed_result, fn_name
        )

        # Phase 2: Generate variants
        valid_programs, behavior_vectors = await self._generate_variants(
            diverse_seeds, fn_name, extractor
        )

        # Phase 3: Build centroids and populate archive
        await self._populate_archive(
            pool, valid_programs, behavior_vectors, extractor,
            seed_program, seed_result  # Always pass seed as fallback
        )

        return self.total_cost

    async def _generate_diverse_seeds(
        self,
        seed_program: str,
        seed_score: float,
        seed_result: dict,
        fn_name: str,
    ) -> list[tuple[str, float, dict]]:
        """Phase 1: Generate diverse seeds sequentially with context accumulation."""
        n_seeds = self.config.init.n_diverse_seeds
        model = self.config.init.diversity_model or "gpt-4"

        logger.info(f"[Init Phase 1] Generating {n_seeds} diverse seeds with {model}")

        # Store (code, score, full_result) tuples - include full result to avoid re-evaluation
        diverse_seeds = [(seed_program, seed_score, seed_result)]

        for i in range(n_seeds):
            existing_seeds_text = "\n\n---\n\n".join([
                f"### Seed {j+1} (Score: {score:.1f}):\n```python\n{code}\n```"
                for j, (code, score, _) in enumerate(diverse_seeds)
            ])

            prompt = DIVERSITY_SEED_PROMPT.format(
                problem_title="Algorithm Optimization",
                problem_description=self.config.problem_description,
                function_signature=self.config.function_signature,
                existing_seeds=existing_seeds_text,
            )

            try:
                llm = get_llm_client()
                response = await llm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.init.temperature,
                    max_tokens=4096,
                    timeout=300,
                )
                content = response.content
                self.total_cost += response.cost

                new_code = extract_code(content)
                if not new_code:
                    logger.info(f"  [Seed {i+1}] Failed to extract code")
                    continue

                result = await self.executor.run(
                    _evaluate_code,
                    new_code,
                    self.config.score_fn,
                    self.config.inputs,
                    fn_name,
                    timeout=self.config.pipeline.eval_timeout
                )

                if "error" not in result:
                    new_score = result.get('score', 0.0)
                    if new_score > self.best_score:
                        self.best_score = new_score
                    diverse_seeds.append((new_code, new_score, result))
                    logger.info(f"  [Seed {i+1}] OK - score: {new_score:.1f}")
                else:
                    logger.info(f"  [Seed {i+1}] Eval failed: {result['error'][:50]}")

            except Exception as e:
                logger.warning(f"  [Seed {i+1}] Error: {e}")

        logger.info(f"[Init Phase 1] Generated {len(diverse_seeds)-1} new seeds (total: {len(diverse_seeds)})")
        return diverse_seeds

    async def _generate_variants(
        self,
        diverse_seeds: list[tuple[str, float, dict]],
        fn_name: str,
        extractor: BehaviorExtractor,
    ) -> tuple[list[dict], list[np.ndarray]]:
        """Phase 2: Generate variants in parallel using light model(s)."""
        n_variants_per_seed = self.config.init.n_variants_per_seed
        models = self.config.init.variant_models or ["gpt-4o-mini"]
        n_variants = n_variants_per_seed * len(diverse_seeds)

        logger.info(f"[Init Phase 2] Generating {n_variants} variants with {models}")

        # Build all seeds as ProgramWithScore for sampling
        all_parents = []
        for seed_code, seed_score, _ in diverse_seeds:
            prog = Program(code=seed_code, metadata={"score": seed_score})
            eval_res = EvaluationResult(
                scores={'score': seed_score},
                is_valid=True,
            )
            all_parents.append(ProgramWithScore(prog, eval_res))

        # Build prompts - each variant randomly samples 2 seeds as inspirations
        prompts = []
        n_inspirations = min(2, len(all_parents))  # Sample 2 seeds as inspirations

        for seed_idx, (seed_code, seed_score, _) in enumerate(diverse_seeds):
            for _ in range(n_variants_per_seed):
                # Randomly sample inspirations from all seeds
                inspirations = random.sample(all_parents, n_inspirations)

                builder = PromptBuilder()
                builder.add_section("Problem", self.config.problem_description, priority=10)
                builder.add_section("Signature", f"```python\n{self.config.function_signature}\n```", priority=20)
                builder.add_parents(inspirations, priority=30)
                builder.set_output_mode(OutputMode.FULL)
                prompts.append(builder.build())

        # Parallel LLM calls (cycle through models)
        async def generate_variant(idx: int, prompt: str, model: str) -> dict:
            try:
                llm = get_llm_client()
                response = await llm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.init.temperature,
                    max_tokens=4096,
                    timeout=300,
                )
                return {"idx": idx, "content": response.content, "cost": response.cost}
            except Exception as e:
                return {"idx": idx, "error": str(e)}

        tasks = [generate_variant(i, prompts[i], models[i % len(models)]) for i in range(n_variants)]
        results = await asyncio.gather(*tasks)

        # Extract code from responses
        candidates = []
        for res in results:
            if "error" in res:
                continue
            self.total_cost += res["cost"]
            code = extract_code(res["content"])
            if code:
                candidates.append({"idx": res["idx"], "code": code})

        total_candidates = len(candidates)
        print(f"[Init Phase 2] Evaluating: 0/{total_candidates}", end='', flush=True)

        # Parallel evaluation with progress logging
        eval_count = 0

        async def eval_candidate(cand: dict) -> tuple[int, dict]:
            nonlocal eval_count
            try:
                result = await self._cascade_eval(cand["code"], fn_name)
                return cand["idx"], {"code": cand["code"], "result": result}
            except Exception as e:
                return cand["idx"], {"code": cand["code"], "result": {"error": str(e)}}
            finally:
                eval_count += 1
                print(f"\r[Init Phase 2] Evaluating: {eval_count}/{total_candidates}", end='', flush=True)

        eval_tasks = [eval_candidate(c) for c in candidates]
        eval_results = await asyncio.gather(*eval_tasks)
        print()  # Newline after progress complete

        # Collect valid programs and behavior vectors
        valid_programs = []
        behavior_vectors = []
        failure_errors = []  # Track unique failure reasons

        for _, data in eval_results:
            result = data["result"]
            if "error" in result:
                # Collect unique failure errors for logging
                error_msg = result["error"]
                if len(failure_errors) < 5 and error_msg not in [e for e, _ in failure_errors]:
                    failure_errors.append((error_msg, data["code"][:100]))
                continue
            score = result.get('score', 0.0)
            program = Program(code=data["code"], metadata={"primary_score": score})
            behavior = extractor.extract(program, result)
            valid_programs.append({
                "code": data["code"],
                "score": score,
                "result": result,
                "behavior": behavior,
            })
            behavior_vectors.append(np.array([behavior[f] for f in extractor.features]))

        # Add diverse seeds directly (already evaluated in Phase 1, no need to re-evaluate)
        for seed_code, seed_score, seed_result in diverse_seeds:
            program = Program(code=seed_code, metadata={"primary_score": seed_score})
            behavior = extractor.extract(program, seed_result)
            valid_programs.append({
                "code": seed_code,
                "score": seed_score,
                "result": seed_result,
                "behavior": behavior,
            })
            behavior_vectors.append(np.array([behavior[f] for f in extractor.features]))

        logger.info(f"[Init Phase 2] Valid programs: {len(valid_programs)}")

        # Log sample of failure reasons if there were failures
        if failure_errors:
            logger.info(f"[Init Phase 2] Sample failures ({len(failure_errors)} unique errors shown):")
            for error_msg, code_preview in failure_errors:
                logger.info(f"  - {error_msg[:100]}")
                logger.info(f"    Code: {code_preview}...")
        return valid_programs, behavior_vectors

    async def _populate_archive(
        self,
        pool: CVTMAPElitesPool,
        valid_programs: list[dict],
        behavior_vectors: list[np.ndarray],
        extractor: BehaviorExtractor,
        seed_program: str = None,
        seed_result: dict = None,
    ) -> None:
        """Phase 3: Build centroids and populate archive using k-means labels directly."""
        # Check for predefined centroids file
        if self.config.cvt.predefined_centroids_file:
            logger.info(f"[Init Phase 3] Loading predefined centroids from {self.config.cvt.predefined_centroids_file}")
            result = _load_predefined_centroids(
                self.config.cvt.predefined_centroids_file,
                extractor.features,
            )

            # Set up extractor normalization
            if result.bounds is not None:
                # Deterministic mode: use fixed bounds from centroids file
                extractor.set_fixed_bounds(result.bounds)
                logger.info(f"[Init Phase 3] Using deterministic normalization with fixed bounds")
            else:
                # Legacy mode: use z-score stats from centroid data
                extractor.init_stats_from_data(result.raw_feature_data)
                logger.info(f"[Init Phase 3] Using adaptive normalization (legacy mode)")

            pool._centroids = result.centroids
            pool._n_centroids = len(result.centroids)
            pool._mins = np.zeros(len(extractor.features))
            pool._maxs = np.ones(len(extractor.features))
            pool._ranges = np.ones(len(extractor.features))
            logger.info(f"[Init Phase 3] Loaded {len(result.centroids)} predefined centroids")

            # Add all valid programs from init phases to the archive
            n_accepted = 0
            for prog_data in valid_programs:
                program = Program(code=prog_data["code"], metadata={"primary_score": prog_data["score"]})
                eval_result = EvaluationResult(
                    scores=prog_data["result"],
                    is_valid=True,
                )
                accepted, _ = pool.add(program, eval_result)
                if accepted:
                    n_accepted += 1

            best_score = max(p["score"] for p in valid_programs) if valid_programs else 0.0
            logger.info(f"[Init Phase 3] Done: {n_accepted} cells filled, archive size: {pool.size()}, best: {best_score:.1f}, cost: ${self.total_cost:.3f}")

            extractor.set_phase('evolution')
            return

        if len(behavior_vectors) < 3:
            logger.warning("[Init Phase 3] Not enough valid programs to build centroids")
            if seed_program and seed_result and "error" not in seed_result:
                program = Program(code=seed_program, metadata={"primary_score": seed_result.get('score', 0.0)})
                eval_result = EvaluationResult(
                    scores=seed_result,
                    is_valid=True,
                )
                accepted, _ = pool.add(program, eval_result)
                logger.info(f"[Init Phase 3] Added seed program as fallback (accepted={accepted}), archive size: {pool.size()}")
            extractor.set_phase('evolution')
            return

        # Build centroids and get k-means labels for each program
        n_centroids, labels = pool.set_centroids_from_data(
            behavior_vectors,
            n_centroids=self.config.cvt.n_centroids,
        )
        logger.info(f"[Init Phase 3] Built {n_centroids} centroids")

        # Group programs by their assigned cell (from k-means labels)
        cell_to_programs: dict[int, list[dict]] = {}
        for idx, prog in enumerate(valid_programs):
            cell = int(labels[idx])
            if cell not in cell_to_programs:
                cell_to_programs[cell] = []
            cell_to_programs[cell].append(prog)

        # For each cell, add the best-scoring program directly (no re-extraction)
        n_accepted = 0
        for cell_idx, progs in cell_to_programs.items():
            best_prog = max(progs, key=lambda x: x["score"])
            program = Program(code=best_prog["code"], metadata={"primary_score": best_prog["score"]})
            eval_result = EvaluationResult(
                scores=best_prog["result"],
                is_valid=True,
            )
            if pool.add_at_cell(cell_idx, program, eval_result, best_prog["behavior"]):
                n_accepted += 1

        best_score = max(p["score"] for p in valid_programs) if valid_programs else 0.0
        logger.info(f"[Init Phase 3] Done: {n_accepted} cells filled, archive size: {pool.size()}, best: {best_score:.1f}, cost: ${self.total_cost:.3f}")

        # Switch extractor to evolution phase
        extractor.set_phase('evolution')
