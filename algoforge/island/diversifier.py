"""
Island Diversifier: generates diverse seeds and distributes to islands.

Flow:
1. Phase 1: Generate diverse seeds sequentially (like single-island)
2. Phase 2: Generate variants for all seeds (30 per seed)
3. Phase 3: Round-robin distribute programs to islands
4. Phase 4: Each island computes its own centroids independently
"""

import asyncio
import logging
import random
import types
from typing import Optional

import litellm
import numpy as np

from ..config import AlgoforgeConfig
from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..behavior import BehaviorExtractor
from ..llm import PromptBuilder, ProgramWithScore, OutputMode
from ..utils import ResilientProcessPool, extract_code, extract_fn_name
from .coordinator import IslandCoordinator

logger = logging.getLogger(__name__)


DIVERSITY_SEED_PROMPT = """{problem_description}

## Function Signature
```python
{function_signature}
```

## Your Task: ALGORITHMIC DIVERSITY

Design a solution using a **FUNDAMENTALLY DIFFERENT ALGORITHM** than the existing seeds.

**DO NOT:**
- Make minor variations or parameter tweaks
- Use the same core algorithm with different constants

**DO:**
- Analyze what paradigm each existing seed uses
- Design from first principles using a different strategy
- Consider what information the existing seeds are NOT using

## Existing Seeds:
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

    fn = namespace.get(fn_name)
    if not isinstance(fn, types.FunctionType):
        return {"error": f"Function '{fn_name}' not found (got {type(fn).__name__})"}

    return score_fn(fn, inputs)


class IslandDiversifier:
    """
    Generates diverse seeds and distributes variants across islands.

    Unlike per-island seeding, this follows the single-island pattern:
    1. Generate diverse seeds with cross-inspiration
    2. Generate variants for all seeds together
    3. Round-robin distribute to islands
    4. Each island builds its own centroids
    """

    def __init__(
        self,
        config: AlgoforgeConfig,
        executor: ResilientProcessPool,
    ):
        self.config = config
        self.executor = executor
        self.total_cost = 0.0

    async def initialize_all_islands(
        self,
        coordinator: IslandCoordinator,
        base_seed: str,
        base_result: dict,
        extractor: BehaviorExtractor,
    ) -> float:
        """
        Initialize all islands with diverse programs.

        1. Generate diverse seeds (sequential with context)
        2. Generate variants for all seeds
        3. Round-robin distribute to islands
        4. Each island computes its own centroids
        """
        fn_name = extract_fn_name(self.config.function_signature)
        seed_score = base_result.get('score', 0.0)
        n_islands = coordinator.n_islands

        # Phase 1: Generate diverse seeds
        diverse_seeds = await self._generate_diverse_seeds(
            base_seed, seed_score, fn_name
        )

        # Phase 2: Generate variants for all seeds
        valid_programs, behavior_vectors = await self._generate_variants(
            diverse_seeds, fn_name, extractor
        )

        if len(valid_programs) < n_islands:
            logger.warning(
                f"[IslandDiversifier] Only {len(valid_programs)} valid programs, "
                f"need at least {n_islands}"
            )
            return self.total_cost

        # Phase 3: Round-robin distribute to islands
        island_programs = [[] for _ in range(n_islands)]
        island_behaviors = [[] for _ in range(n_islands)]

        # Sort by score descending before distribution
        sorted_indices = sorted(
            range(len(valid_programs)),
            key=lambda i: valid_programs[i]["score"],
            reverse=True
        )

        for rank, idx in enumerate(sorted_indices):
            island_idx = rank % n_islands
            island_programs[island_idx].append(valid_programs[idx])
            island_behaviors[island_idx].append(behavior_vectors[idx])

        # Phase 4: Initialize each island with its programs
        for i in range(n_islands):
            await self._initialize_island(
                i, coordinator, island_programs[i], island_behaviors[i]
            )

        # Switch to evolution phase
        extractor.set_phase('evolution')

        logger.info(
            f"[IslandDiversifier] All {n_islands} islands initialized, "
            f"total cost: ${self.total_cost:.3f}"
        )
        return self.total_cost

    async def _generate_diverse_seeds(
        self,
        seed_program: str,
        seed_score: float,
        fn_name: str,
    ) -> list[tuple[str, float]]:
        """Phase 1: Generate diverse seeds sequentially with context accumulation."""
        n_seeds = self.config.init.n_diverse_seeds
        model = self.config.init.diversity_model or "gpt-4"

        logger.info(f"[IslandDiversifier] Phase 1: Generating {n_seeds} diverse seeds with {model}")

        diverse_seeds = [(seed_program, seed_score)]

        for i in range(n_seeds):
            existing_seeds_text = "\n\n---\n\n".join([
                f"### Seed {j+1} (Score: {score:.1f}):\n```python\n{code}\n```"
                for j, (code, score) in enumerate(diverse_seeds)
            ])

            prompt = DIVERSITY_SEED_PROMPT.format(
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
                    logger.warning(f"  [Seed {i+1}] Failed to extract code")
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
                    diverse_seeds.append((new_code, new_score))
                    logger.info(f"  [Seed {i+1}] OK - score: {new_score:.1f}")
                else:
                    logger.warning(f"  [Seed {i+1}] Failed: {result['error'][:80]}")

            except Exception as e:
                logger.warning(f"  [Seed {i+1}] Error: {e}")

        logger.info(
            f"[IslandDiversifier] Phase 1 complete: {len(diverse_seeds)} seeds "
            f"(1 base + {len(diverse_seeds)-1} new)"
        )
        return diverse_seeds

    async def _generate_variants(
        self,
        diverse_seeds: list[tuple[str, float]],
        fn_name: str,
        extractor: BehaviorExtractor,
    ) -> tuple[list[dict], list[np.ndarray]]:
        """Phase 2: Generate variants for all seeds."""
        n_variants_per_seed = self.config.init.n_variants_per_seed
        model = self.config.init.variant_model or "gpt-4o-mini"
        total_variants = n_variants_per_seed * len(diverse_seeds)

        logger.info(
            f"[IslandDiversifier] Phase 2: Generating {total_variants} variants "
            f"({n_variants_per_seed} per seed) with {model}"
        )

        # Build seed programs for prompt building
        seed_programs = []
        for seed_code, seed_score in diverse_seeds:
            prog = Program(code=seed_code, metadata={"score": seed_score})
            eval_res = EvaluationResult(
                program_id=prog.id,
                scores={'score': seed_score},
                is_valid=True,
            )
            seed_programs.append(ProgramWithScore(prog, eval_res))

        # Build prompts with cross-inspiration
        prompts = []
        n_inspirations = self.config.pipeline.n_inspirations

        for seed_idx, (seed_code, seed_score) in enumerate(diverse_seeds):
            parent = seed_programs[seed_idx]
            other_seeds = [sp for i, sp in enumerate(seed_programs) if i != seed_idx]

            for _ in range(n_variants_per_seed):
                inspirations = (
                    random.sample(other_seeds, min(n_inspirations, len(other_seeds)))
                    if other_seeds and n_inspirations > 0
                    else []
                )

                builder = PromptBuilder()
                builder.add_section("Problem", self.config.problem_description, priority=10)
                builder.add_section(
                    "Signature",
                    f"```python\n{self.config.function_signature}\n```",
                    priority=20
                )
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

        tasks = [generate_variant(i, prompts[i]) for i in range(total_variants)]
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

        n_candidates = len(candidates)
        logger.info(f"[IslandDiversifier] Evaluating {n_candidates} candidates")

        # Parallel evaluation with progress
        async def eval_candidate(cand: dict) -> tuple[int, dict]:
            try:
                result = await self.executor.run(
                    _evaluate_code,
                    cand["code"],
                    self.config.score_fn,
                    self.config.inputs,
                    fn_name,
                    timeout=self.config.pipeline.eval_timeout
                )
                return cand["idx"], {"code": cand["code"], "result": result}
            except Exception as e:
                return cand["idx"], {"code": cand["code"], "result": {"error": str(e)}}

        eval_tasks = [eval_candidate(c) for c in candidates]

        # Use as_completed for progress updates
        eval_results = []
        completed = 0
        valid = 0
        for coro in asyncio.as_completed(eval_tasks):
            result = await coro
            eval_results.append(result)
            completed += 1
            if "error" not in result[1]["result"]:
                valid += 1
            # Update progress every 10 or at end
            if completed % 10 == 0 or completed == n_candidates:
                print(f"\r  Evaluated: {completed}/{n_candidates} ({valid} valid)", end="", flush=True)
        print()  # newline after progress

        # Collect valid programs and behavior vectors
        valid_programs = []
        behavior_vectors = []

        extractor.set_phase('init')

        for _, data in eval_results:
            result = data["result"]
            if "error" in result:
                continue
            score = result.get('score', 0.0)
            program = Program(code=data["code"], metadata={"primary_score": score})
            behavior = extractor.extract(program, result)
            valid_programs.append({
                "code": data["code"],
                "score": score,
                "result": result,
            })
            behavior_vectors.append(
                np.array([behavior[f] for f in extractor.features])
            )

        # Also add diverse seeds themselves
        for seed_code, seed_score in diverse_seeds:
            try:
                result = await self.executor.run(
                    _evaluate_code,
                    seed_code,
                    self.config.score_fn,
                    self.config.inputs,
                    fn_name,
                    timeout=self.config.pipeline.eval_timeout
                )
                if "error" not in result:
                    program = Program(code=seed_code, metadata={"primary_score": seed_score})
                    behavior = extractor.extract(program, result)
                    valid_programs.append({
                        "code": seed_code,
                        "score": seed_score,
                        "result": result,
                    })
                    behavior_vectors.append(
                        np.array([behavior[f] for f in extractor.features])
                    )
            except Exception:
                pass

        logger.info(f"[IslandDiversifier] Phase 2 complete: {len(valid_programs)} valid programs")
        return valid_programs, behavior_vectors

    async def _initialize_island(
        self,
        island_idx: int,
        coordinator: IslandCoordinator,
        programs: list[dict],
        behaviors: list[np.ndarray],
    ) -> None:
        """Phase 4: Initialize a single island with its assigned programs."""
        island = coordinator.get_island(island_idx)
        pool = island.pool

        if len(behaviors) < 3:
            logger.warning(f"[Island {island_idx}] Not enough programs to build centroids")
            return

        # Build centroids from this island's behavior data (with its own noise)
        n_centroids = pool.set_centroids_from_data(
            behaviors,
            percentile_low=5.0,
            percentile_high=95.0,
            n_centroids=coordinator.n_centroids,
        )

        # Add programs to archive
        n_accepted = 0
        for prog in programs[:coordinator.n_centroids]:
            program = Program(code=prog["code"], metadata={"primary_score": prog["score"]})
            eval_result = EvaluationResult(
                program_id=program.id,
                scores=prog["result"],
                is_valid=True,
            )
            if pool.add(program, eval_result):
                n_accepted += 1
                island.acceptance_count += 1

        best_score = programs[0]["score"] if programs else 0.0
        logger.info(
            f"[Island {island_idx}] Initialized: {n_centroids} centroids, "
            f"{n_accepted} accepted, best: {best_score:.1f}"
        )

    # Keep legacy methods for backward compatibility
    async def generate_island_seeds(self, *args, **kwargs):
        """Deprecated: use initialize_all_islands instead."""
        raise NotImplementedError("Use initialize_all_islands instead")

    async def initialize_island(self, *args, **kwargs):
        """Deprecated: use initialize_all_islands instead."""
        raise NotImplementedError("Use initialize_all_islands instead")
