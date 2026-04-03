"""Init phase: diverse seed generation and archive initialization."""

import asyncio
import logging
import random

import numpy as np

from ..behavior import BehaviorExtractor
from ..config import LeviConfig
from ..core import EvaluationResult, Program
from ..llm import OutputMode, ProgramWithScore, PromptBuilder, get_llm_client
from ..pipeline.state import BudgetLimitReached, PipelineState, ScoreHistoryEntry
from ..pool import CVTMAPElitesPool
from ..utils import ResilientProcessPool, evaluate_code, extract_code, extract_fn_name

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


class Diversifier:
    """Handles init phase: diverse seed generation and archive population."""

    def __init__(
        self,
        config: LeviConfig,
        executor: ResilientProcessPool,
        state: PipelineState,
    ):
        self.config = config
        self.executor = executor
        self.state = state
        self.total_cost = 0.0
        self.best_score = 0.0

    async def _acompletion(self, *, model: str, messages: list, **kwargs):
        """Route completion through state for budget tracking."""
        llm = get_llm_client()
        return await self.state.acompletion(llm, model=model, messages=messages, **kwargs)

    def _record_score(self, score: float, sampler: str) -> None:
        """Record a score from the init phase."""
        self.state.record_accept()
        self.state.record_score(
            score=score,
            accepted=True,
            sampler=sampler,
            archive_size=0,
            cell_index=None,
        )
        self.best_score = self.state.best_score_so_far

    async def _cascade_eval(self, code: str, fn_name: str) -> dict:
        """Evaluate with cascade: quick test first, full eval only if promising."""
        # Caller already reserved the eval slot.

        cascade = self.config.cascade
        if cascade.enabled and cascade.quick_inputs:
            quick_result = await self.executor.run(
                evaluate_code, code, self.config.score_fn, cascade.quick_inputs, fn_name, timeout=cascade.quick_timeout
            )
            if "error" in quick_result:
                return quick_result
            quick_score = quick_result.get("score", 0)
            threshold = self.best_score * cascade.min_score_ratio
            if quick_score < threshold:
                return {"error": f"Cascade rejected: {quick_score:.17g} < {threshold:.17g}"}

        result = await self.executor.run(
            evaluate_code,
            code,
            self.config.score_fn,
            self.config.inputs,
            fn_name,
            timeout=self.config.pipeline.eval_timeout,
        )
        if "error" not in result:
            score = result.get("score", 0)
            self._record_score(score, sampler="init_variant")
        return result

    async def run(
        self,
        pool: CVTMAPElitesPool,
        seed_program: str | None,
        seed_result: dict | None,
        extractor: BehaviorExtractor,
    ) -> tuple[float, list[ScoreHistoryEntry]]:
        """
        Run init phase: generate diverse seeds, expand with variants, build centroids.

        Returns (total_cost, score_history) from the init phase.
        """
        fn_name = extract_fn_name(self.config.function_signature)

        if seed_result is not None:
            seed_score = seed_result.get("score", 0.0)
            self.best_score = seed_score
            # Record the seed itself
            if self.state.eval_count == 0:
                self._record_score(seed_score, sampler="init_seed")

        # Phase 1: Generate diverse seeds
        diverse_seeds = await self._generate_diverse_seeds(seed_program, seed_result, fn_name)

        # Phase 2: Generate variants
        valid_programs, behavior_vectors = await self._generate_variants(diverse_seeds, fn_name, extractor)

        # Phase 3: Build centroids and populate archive
        await self._populate_archive(
            pool,
            valid_programs,
            behavior_vectors,
            extractor,
            seed_program,
            seed_result,  # Always pass seed as fallback
        )

        return self.total_cost, list(self.state.score_history)

    async def _generate_diverse_seeds(
        self,
        seed_program: str | None,
        seed_result: dict | None,
        fn_name: str,
    ) -> list[tuple[str, float, dict]]:
        """Phase 1: Generate diverse seeds sequentially with context accumulation."""
        n_seeds = self.config.init.n_diverse_seeds
        model = self.config.init.diversity_model or "gpt-4"

        # Store (code, score, full_result) tuples - include full result to avoid re-evaluation
        diverse_seeds: list[tuple[str, float, dict]] = []
        if seed_program is not None and seed_result is not None:
            diverse_seeds.append((seed_program, seed_result.get("score", 0.0), seed_result))
        else:
            # Generate more seeds to compensate for missing seed program
            n_seeds += 1

        logger.info(f"[Init Phase 1] Generating {n_seeds} diverse seeds with {model}")

        max_retries = 3
        for i in range(n_seeds):
            if self.state.budget_exhausted:
                logger.info("[Init Phase 1] Stopping seed generation (budget exhausted)")
                break

            success = False
            for attempt in range(max_retries):
                if self.state.budget_exhausted:
                    break

                attempt_label = f"[Seed {i + 1}]" if attempt == 0 else f"[Seed {i + 1}, retry {attempt}]"

                existing_seeds_text = "\n\n---\n\n".join(
                    [
                        f"### Seed {j + 1} (Score: {score:.17g}):\n```python\n{code}\n```"
                        for j, (code, score, _) in enumerate(diverse_seeds)
                    ]
                )

                # Use custom prompt if provided, otherwise use default
                prompt_template = self.config.init.diversity_prompt or DIVERSITY_SEED_PROMPT
                prompt = prompt_template.format(
                    problem_title="Algorithm Optimization",
                    problem_description=self.config.problem_description,
                    function_signature=self.config.function_signature,
                    existing_seeds=existing_seeds_text,
                )

                try:
                    llm_kwargs = {
                        "max_tokens": 16384,
                        "timeout": 300,
                        **self.config.init.diversity_llm_kwargs,
                    }
                    response = await self._acompletion(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.init.temperature,
                        **llm_kwargs,
                    )
                    content = response.content
                    self.total_cost += response.cost

                    new_code = extract_code(content)
                    if not new_code:
                        logger.info(f"  {attempt_label} Failed to extract code")
                        continue

                    if not await self.state.try_start_evaluation():
                        logger.info("[Init Phase 1] Stopping seed evaluation (budget exhausted)")
                        break

                    try:
                        result = await self.executor.run(
                            evaluate_code,
                            new_code,
                            self.config.score_fn,
                            self.config.inputs,
                            fn_name,
                            timeout=self.config.pipeline.eval_timeout,
                        )

                        if "error" not in result:
                            new_score = result.get("score", 0.0)
                            self._record_score(new_score, sampler="init_diversity")
                            diverse_seeds.append((new_code, new_score, result))
                            logger.info(f"  {attempt_label} OK - score: {new_score:.17g}")
                            success = True
                        else:
                            self.state.record_error(str(result.get("error", "unknown error")))
                            logger.info(f"  {attempt_label} Eval failed: {result['error'][:50]}")
                    finally:
                        await self.state.finish_evaluation()
                except BudgetLimitReached:
                    logger.info("[Init Phase 1] Stopping seed generation (budget exhausted)")
                    break
                except Exception as e:
                    logger.warning(f"  {attempt_label} Error: {e}")

                if success:
                    break

        logger.info(f"[Init Phase 1] Generated {len(diverse_seeds)} seeds")
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
            prog = Program(content=seed_code, metadata={"score": seed_score})
            eval_res = EvaluationResult(
                scores={"score": seed_score},
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
                response = await self._acompletion(
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
            if not await self.state.try_start_evaluation():
                return cand["idx"], {
                    "code": cand["code"],
                    "result": {"error": "Budget exhausted", "budget_stop": True},
                }
            try:
                result = await self._cascade_eval(cand["code"], fn_name)
                if "error" in result:
                    error_msg = str(result.get("error", "unknown error"))
                    if error_msg.startswith("Cascade rejected:"):
                        self.state.record_reject()
                    elif error_msg != "Budget exhausted":
                        self.state.record_error(error_msg)
                return cand["idx"], {"code": cand["code"], "result": result}
            except Exception as e:
                self.state.record_error(str(e))
                return cand["idx"], {"code": cand["code"], "result": {"error": str(e)}}
            finally:
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
            score = result.get("score", 0.0)
            program = Program(content=data["code"], metadata={"primary_score": score})
            behavior = extractor.extract(program, result)
            valid_programs.append(
                {
                    "code": data["code"],
                    "score": score,
                    "result": result,
                    "behavior": behavior,
                }
            )
            behavior_vectors.append(np.array([behavior[f] for f in extractor.features]))

        # Add diverse seeds directly (already evaluated in Phase 1, no need to re-evaluate)
        for seed_code, seed_score, seed_result in diverse_seeds:
            program = Program(content=seed_code, metadata={"primary_score": seed_score})
            behavior = extractor.extract(program, seed_result)
            valid_programs.append(
                {
                    "code": seed_code,
                    "score": seed_score,
                    "result": seed_result,
                    "behavior": behavior,
                }
            )
            behavior_vectors.append(np.array([behavior[f] for f in extractor.features]))

        logger.info(f"[Init Phase 2] Valid programs: {len(valid_programs)}")

        # Log sample of failure reasons if there were failures
        if failure_errors:
            logger.info(f"[Init Phase 2] Sample failures ({len(failure_errors)} unique errors shown):")
            for error_msg, code_preview in failure_errors:
                logger.info(f"  - {error_msg[:100]}")
                logger.info(f"    Code: {code_preview}...")
        return valid_programs, behavior_vectors

    async def _backfill_quick_scores(self, programs: list[dict]) -> None:
        """Compute quick_scores for programs that don't have one yet.

        Modifies each program's 'result' dict in-place, adding 'quick_score'.
        Programs that already have a quick_score or where quick eval fails are skipped.
        """
        cascade = self.config.cascade
        if not (cascade.enabled and cascade.quick_inputs):
            return

        fn_name = extract_fn_name(self.config.function_signature)
        tasks = []
        indices = []
        for i, prog in enumerate(programs):
            if "quick_score" not in prog["result"]:
                tasks.append(
                    self.executor.run(
                        evaluate_code,
                        prog["code"],
                        self.config.score_fn,
                        cascade.quick_inputs,
                        fn_name,
                        timeout=cascade.quick_timeout,
                    )
                )
                indices.append(i)

        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, qr in zip(indices, results):
            if isinstance(qr, Exception) or "error" in qr:
                continue
            programs[i]["result"]["quick_score"] = qr.get("score", 0)

        logger.info(f"[Init Phase 3] Computed quick_scores for {len(tasks)} archive seeds")

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
                fallback = [{"code": seed_program, "result": {**seed_result}}]
                await self._backfill_quick_scores(fallback)

                program = Program(content=seed_program, metadata={"primary_score": seed_result.get("score", 0.0)})
                eval_result = EvaluationResult(
                    scores=fallback[0]["result"],
                    is_valid=True,
                )
                accepted, _ = pool.add(program, eval_result)
                logger.info(
                    f"[Init Phase 3] Added seed program as fallback (accepted={accepted}), archive size: {pool.size()}"
                )
            extractor.set_phase("evolution")
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

        # Select best program per cell and backfill quick_scores for cascade
        best_per_cell = {
            cell_idx: max(progs, key=lambda p: p["score"])
            for cell_idx, progs in cell_to_programs.items()
        }
        await self._backfill_quick_scores(list(best_per_cell.values()))

        # For each cell, add the best-scoring program directly (no re-extraction)
        n_accepted = 0
        for cell_idx, best_prog in best_per_cell.items():
            program = Program(content=best_prog["code"], metadata={"primary_score": best_prog["score"]})
            eval_result = EvaluationResult(
                scores=best_prog["result"],
                is_valid=True,
            )
            if pool.add_at_cell(cell_idx, program, eval_result, best_prog["behavior"]):
                n_accepted += 1

        best_score = max(p["score"] for p in valid_programs) if valid_programs else 0.0
        logger.info(
            f"[Init Phase 3] Done: {n_accepted} cells filled, archive size: {pool.size()}, best: {best_score:.17g}, cost: ${self.total_cost:.3f}"
        )

        # Switch extractor to evolution phase
        extractor.set_phase("evolution")
