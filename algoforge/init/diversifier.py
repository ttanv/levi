"""Init phase: diverse seed generation and archive initialization."""

import asyncio
import logging
import random
import time
import types

import numpy as np

from ..config import AlgoforgeConfig
from ..core import Program, EvaluationResult
from ..pipeline.state import ScoreHistoryEntry, PipelineState, BudgetLimitReached
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


class Diversifier:
    """Handles init phase: diverse seed generation and archive population."""

    def __init__(
        self,
        config: AlgoforgeConfig,
        executor: ResilientProcessPool,
        start_time: float = 0.0,
        state: PipelineState | None = None,
    ):
        self.config = config
        self.executor = executor
        self.state = state
        self.total_cost = 0.0
        self.best_score = 0.0
        self.score_history: list[ScoreHistoryEntry] = []
        self._init_eval_count = 0
        self._start_time = start_time or time.time()

    def _record_score(self, score: float, sampler: str) -> None:
        """Record a score from the init phase."""
        if self.state is not None:
            self.state.record_accept()
            self.state.record_score(
                score=score,
                accepted=True,
                sampler=sampler,
                archive_size=0,
                cell_index=None,
            )
            self.best_score = self.state.best_score_so_far
            return

        self._init_eval_count += 1
        if score > self.best_score:
            self.best_score = score
        self.score_history.append(ScoreHistoryEntry(
            eval_number=self._init_eval_count,
            score=score,
            best_score=self.best_score,
            timestamp=time.time() - self._start_time,
            accepted=True,  # All valid init programs go into the archive
            sampler=sampler,
            archive_size=0,  # Not meaningful during init
            cell_index=None,
            cumulative_cost=self.total_cost,
        ))

    async def _cascade_eval(self, code: str, fn_name: str) -> dict:
        """Evaluate with cascade: quick test first, full eval only if promising."""
        if self.state is not None and self.state.budget_exhausted:
            return {"error": "Budget exhausted"}

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
            self._record_score(score, sampler="init_variant")
        return result

    async def run(
        self,
        pool: CVTMAPElitesPool,
        seed_program: str,
        seed_result: dict,
        extractor: BehaviorExtractor,
    ) -> tuple[float, list[ScoreHistoryEntry]]:
        """
        Run init phase: generate diverse seeds, expand with variants, build centroids.

        Returns (total_cost, score_history) from the init phase.
        """
        fn_name = extract_fn_name(self.config.function_signature)
        seed_score = seed_result.get('score', 0.0)
        self.best_score = seed_score

        # Record the seed itself
        if self.state is None or self.state.eval_count == 0:
            self._record_score(seed_score, sampler="init_seed")

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

        if self.state is not None:
            return self.total_cost, list(self.state.score_history)
        return self.total_cost, self.score_history

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
            if self.state is not None and self.state.budget_exhausted:
                logger.info("[Init Phase 1] Stopping seed generation (budget exhausted)")
                break

            existing_seeds_text = "\n\n---\n\n".join([
                f"### Seed {j+1} (Score: {score:.1f}):\n```python\n{code}\n```"
                for j, (code, score, _) in enumerate(diverse_seeds)
            ])

            # Use custom prompt if provided, otherwise use default
            prompt_template = self.config.init.diversity_prompt or DIVERSITY_SEED_PROMPT
            prompt = prompt_template.format(
                problem_title="Algorithm Optimization",
                problem_description=self.config.problem_description,
                function_signature=self.config.function_signature,
                existing_seeds=existing_seeds_text,
            )

            try:
                llm = get_llm_client()
                if self.state is not None:
                    response = await self.state.acompletion(
                        llm,
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.init.temperature,
                        max_tokens=4096,
                        timeout=300,
                    )
                else:
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

                reserved_eval = False
                if self.state is not None:
                    if not await self.state.try_start_evaluation():
                        logger.info("[Init Phase 1] Stopping seed evaluation (budget exhausted)")
                        break
                    reserved_eval = True

                try:
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
                        self._record_score(new_score, sampler="init_diversity")
                        diverse_seeds.append((new_code, new_score, result))
                        logger.info(f"  [Seed {i+1}] OK - score: {new_score:.1f}")
                    else:
                        if self.state is not None:
                            self.state.record_error(str(result.get("error", "unknown error")))
                        logger.info(f"  [Seed {i+1}] Eval failed: {result['error'][:50]}")
                finally:
                    if reserved_eval and self.state is not None:
                        await self.state.finish_evaluation()
            except BudgetLimitReached:
                logger.info("[Init Phase 1] Stopping seed generation (budget exhausted)")
                break
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
                if self.state is not None:
                    response = await self.state.acompletion(
                        llm,
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.init.temperature,
                        max_tokens=4096,
                        timeout=300,
                    )
                else:
                    response = await llm.acompletion(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.init.temperature,
                        max_tokens=4096,
                        timeout=300,
                    )
                return {"idx": idx, "content": response.content, "cost": response.cost}
            except BudgetLimitReached:
                return {"idx": idx, "error": "Budget exhausted"}
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
        logger.info(f"[Init Phase 2] Evaluating {total_candidates} candidates")

        async def eval_candidate(cand: dict) -> tuple[int, dict]:
            if self.state is not None:
                if not await self.state.try_start_evaluation():
                    return cand["idx"], {"code": cand["code"], "result": {"error": "Budget exhausted", "budget_stop": True}}
            try:
                result = await self._cascade_eval(cand["code"], fn_name)
                if self.state is not None and "error" in result:
                    error_msg = str(result.get("error", "unknown error"))
                    if error_msg.startswith("Cascade rejected:"):
                        self.state.record_reject()
                    elif error_msg != "Budget exhausted":
                        self.state.record_error(error_msg)
                return cand["idx"], {"code": cand["code"], "result": result}
            except Exception as e:
                if self.state is not None:
                    self.state.record_error(str(e))
                return cand["idx"], {"code": cand["code"], "result": {"error": str(e)}}
            finally:
                if self.state is not None:
                    await self.state.finish_evaluation()

        eval_tasks = [eval_candidate(c) for c in candidates]
        eval_results = await asyncio.gather(*eval_tasks)
        logger.info(f"[Init Phase 2] Finished evaluating {total_candidates} candidates")

        # Collect valid programs and behavior vectors
        valid_programs = []
        behavior_vectors = []
        failure_errors = []  # Track unique failure reasons

        for _, data in eval_results:
            result = data["result"]
            if "error" in result:
                if result.get("budget_stop"):
                    continue
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
        if not behavior_vectors:
            logger.warning("[Init Phase 3] No valid programs to build centroids")
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

        # Build centroids from init-generated programs (seed + diverse seeds + variants).
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

        # For each cell, add a random program directly (no re-extraction)
        n_accepted = 0
        for cell_idx, progs in cell_to_programs.items():
            best_prog = random.choice(progs)
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
