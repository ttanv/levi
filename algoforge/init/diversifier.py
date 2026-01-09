"""Init phase: diverse seed generation and archive initialization."""

import asyncio
import logging
import random
import re
import resource
import types
from typing import Optional

import litellm
import numpy as np

from ..config import AlgoforgeConfig
from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..behavior import BehaviorExtractor
from ..llm import PromptBuilder, ProgramWithScore, OutputMode
from ..utils import ResilientProcessPool

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


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from LLM response."""
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)

    matches = re.findall(r'```python\s*(.*?)```', response, re.DOTALL)
    if matches:
        return matches[0].strip()

    matches = re.findall(r'```\s*(.*?)```', response, re.DOTALL)
    if matches:
        return matches[0].strip()

    return None


def extract_fn_name(fn_signature: str) -> str:
    match = re.search(r'def\s+(\w+)\s*\(', fn_signature)
    if match:
        return match.group(1)
    return 'solve'


def _evaluate_code(code: str, score_fn, inputs: list, fn_name: str) -> dict:
    """Runs in subprocess: parse code, extract callable, call score_fn."""
    # Limit process memory to 2GB to prevent VM crashes
    try:
        memory_bytes = 2 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, resource.error):
        pass  # May fail on some platforms

    namespace = {}
    try:
        exec(code, namespace)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}
    except MemoryError:
        return {"error": "MemoryError: code exceeded 2GB limit"}

    fn = namespace.get(fn_name)
    if not isinstance(fn, types.FunctionType):
        return {"error": f"Function '{fn_name}' not found (got {type(fn).__name__})"}

    return score_fn(fn, inputs)


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
                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.init.temperature,
                    max_tokens=30000,
                    timeout=300,
                )
                content = response.choices[0].message.content
                cost = litellm.completion_cost(completion_response=response)
                self.total_cost += cost

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
        """Phase 2: Generate variants in parallel using light model."""
        n_variants_per_seed = self.config.init.n_variants_per_seed
        model = self.config.init.variant_model or "gpt-4o-mini"
        n_variants = n_variants_per_seed * len(diverse_seeds)

        logger.info(f"[Init Phase 2] Generating {n_variants} variants with {model}")

        # Build prompts for all variants with inspirations from other seeds
        prompts = []
        n_inspirations = self.config.pipeline.n_inspirations

        seed_programs = []
        for seed_code, seed_score, _ in diverse_seeds:
            prog = Program(code=seed_code, metadata={"score": seed_score})
            eval_res = EvaluationResult(
                program_id=prog.id,
                scores={'score': seed_score},
                is_valid=True,
            )
            seed_programs.append(ProgramWithScore(prog, eval_res))

        for seed_idx, (seed_code, seed_score, _) in enumerate(diverse_seeds):
            parent = seed_programs[seed_idx]
            other_seeds = [sp for i, sp in enumerate(seed_programs) if i != seed_idx]

            for _ in range(n_variants_per_seed):
                inspirations = random.sample(other_seeds, min(n_inspirations, len(other_seeds))) if other_seeds and n_inspirations > 0 else []

                builder = PromptBuilder()
                builder.add_section("Problem", self.config.problem_description, priority=10)
                builder.add_section("Signature", f"```python\n{self.config.function_signature}\n```", priority=20)
                builder.add_parents([parent] + inspirations, priority=30)
                builder.set_output_mode(OutputMode.FULL)
                prompts.append(builder.build())

        # Parallel LLM calls
        async def generate_variant(idx: int, prompt: str) -> dict:
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.init.temperature,
                    max_tokens=30000,
                    timeout=300,
                )
                content = response.choices[0].message.content
                cost = litellm.completion_cost(completion_response=response)
                return {"idx": idx, "content": content, "cost": cost}
            except Exception as e:
                return {"idx": idx, "error": str(e)}

        tasks = [generate_variant(i, prompts[i]) for i in range(n_variants)]
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
        """Phase 3: Build centroids and populate archive."""
        if len(behavior_vectors) < 3:
            logger.warning("[Init Phase 3] Not enough valid programs to build centroids")
            # Still add the seed program as fallback so archive isn't empty
            if seed_program and seed_result and "error" not in seed_result:
                program = Program(code=seed_program, metadata={"primary_score": seed_result.get('score', 0.0)})
                eval_result = EvaluationResult(
                    program_id=program.id,
                    scores=seed_result,
                    is_valid=True,
                )
                pool.add(program, eval_result)
                logger.info(f"[Init Phase 3] Added seed program as fallback, archive size: {pool.size()}")
            extractor.set_phase('evolution')
            return

        # Build centroids from behavior data
        n_centroids = pool.set_centroids_from_data(
            behavior_vectors,
            percentile_low=5.0,
            percentile_high=95.0,
            n_centroids=self.config.cvt.n_centroids,
        )
        logger.info(f"[Init Phase 3] Built {n_centroids} centroids")

        # Sort by score and add top programs
        valid_programs.sort(key=lambda x: x["score"], reverse=True)
        top_programs = valid_programs[:self.config.cvt.n_centroids]

        n_accepted = 0
        for prog in top_programs:
            program = Program(code=prog["code"], metadata={"primary_score": prog["score"]})
            eval_result = EvaluationResult(
                program_id=program.id,
                scores=prog["result"],
                is_valid=True,
            )
            if pool.add(program, eval_result):
                n_accepted += 1

        best_score = valid_programs[0]["score"] if valid_programs else 0.0
        logger.info(f"[Init Phase 3] Done: {n_accepted} accepted, archive size: {pool.size()}, best: {best_score:.1f}, cost: ${self.total_cost:.3f}")

        # Switch extractor to evolution phase
        extractor.set_phase('evolution')
