"""
Island Diversifier for generating algorithmically-different seeds per island.

Uses sequential divergence: each island's seed is generated after seeing
all previous islands' seeds, ensuring fundamentally different algorithmic
approaches across islands.
"""

import asyncio
import logging
import random
import re
from typing import Optional

import litellm
import numpy as np

from ..config import AlgoforgeConfig
from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..behavior import BehaviorExtractor
from ..llm import PromptBuilder, ProgramWithScore, OutputMode
from ..utils import ResilientProcessPool
from .coordinator import IslandCoordinator

logger = logging.getLogger(__name__)


ISLAND_SEED_PROMPT = """{problem_description}

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

## Common Mistake
If using functools.reduce, the signature is `reduce(function, iterable)` - function FIRST.

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
    match = re.match(r'def\s+(\w+)\s*\(', fn_signature)
    if match:
        return match.group(1)
    return 'solve'


def _evaluate_code(code: str, score_fn, inputs: list, fn_name: str) -> dict:
    """Runs in subprocess: parse code, extract callable, call score_fn."""
    namespace = {}
    exec(code, namespace)

    fn = namespace.get(fn_name)
    if fn is None:
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith('_'):
                fn = obj
                break

    if fn is None:
        return {"error": f"Function '{fn_name}' not found in generated code"}

    return score_fn(fn, inputs)


class IslandDiversifier:
    """
    Generates algorithmically-different seeds for each island.

    Uses sequential divergence: Island 0 gets the base seed, Island 1
    generates a diverse seed seeing Island 0's, Island 2 sees both, etc.
    Then each island generates its own variants and computes its own
    CVT centroids independently.
    """

    def __init__(
        self,
        config: AlgoforgeConfig,
        executor: ResilientProcessPool,
    ):
        self.config = config
        self.executor = executor
        self.total_cost = 0.0

    async def generate_island_seeds(
        self,
        base_seed: str,
        base_result: dict,
        n_islands: int,
        pool_multiplier: int = 3,
    ) -> list[tuple[str, dict]]:
        """
        Generate a pool of diverse seeds and select the best n_islands.

        Creates n_islands * pool_multiplier candidates, then picks top by score.
        """
        fn_name = extract_fn_name(self.config.function_signature)
        model = self.config.init.diversity_model or "gpt-4"
        n_candidates = n_islands * pool_multiplier

        logger.info(
            f"[IslandDiversifier] Generating {n_candidates} candidates, "
            f"selecting best {n_islands} with {model}"
        )

        pool: list[tuple[str, dict]] = [(base_seed, base_result)]

        for i in range(n_candidates - 1):
            existing_seeds_text = "\n\n---\n\n".join([
                f"### Seed {j+1} (Score: {result.get('score', 0.0):.1f}):\n```python\n{code}\n```"
                for j, (code, result) in enumerate(pool)
            ])

            prompt = ISLAND_SEED_PROMPT.format(
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
                    logger.warning(f"  [Candidate {i+1}] Failed to extract code")
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
                    pool.append((new_code, result))
                    logger.info(
                        f"  [Candidate {i+1}] score: {result.get('score', 0.0):.1f} "
                        f"(pool: {len(pool)})"
                    )
                else:
                    logger.warning(f"  [Candidate {i+1}] Failed: {result['error']}")
                    logger.warning(f"  [Candidate {i+1}] Code:\n{new_code}")

            except Exception as e:
                logger.warning(f"  [Candidate {i+1}] Error: {e}")

        pool.sort(key=lambda x: x[1].get('score', 0.0), reverse=True)
        selected = pool[:n_islands]

        logger.info(
            f"[IslandDiversifier] Selected {len(selected)}/{len(pool)} seeds, "
            f"cost: ${self.total_cost:.3f}"
        )
        for i, (_, result) in enumerate(selected):
            logger.info(f"  Island {i}: score {result.get('score', 0.0):.1f}")

        return selected

    async def initialize_island(
        self,
        island_idx: int,
        coordinator: IslandCoordinator,
        seed_code: str,
        seed_result: dict,
        extractor: BehaviorExtractor,
    ) -> float:
        """
        Initialize a single island with variants of its seed.

        Generates variants, computes CVT centroids, and populates the archive.

        Args:
            island_idx: Index of the island to initialize
            coordinator: The IslandCoordinator
            seed_code: The island's seed program
            seed_result: Evaluation result of the seed
            extractor: Behavior extractor

        Returns:
            Cost incurred during initialization
        """
        island = coordinator.get_island(island_idx)
        pool = island.pool
        fn_name = extract_fn_name(self.config.function_signature)
        seed_score = seed_result.get('score', 0.0)

        logger.info(f"[Island {island_idx}] Initializing with seed score {seed_score:.1f}")

        # Generate variants from the seed
        valid_programs, behavior_vectors = await self._generate_variants(
            seed_code, seed_score, fn_name, extractor, island_idx
        )

        if len(behavior_vectors) < 3:
            logger.warning(
                f"[Island {island_idx}] Not enough valid programs to build centroids"
            )
            return self.total_cost

        # Build centroids from this island's behavior data
        n_centroids = pool.set_centroids_from_data(
            behavior_vectors,
            percentile_low=5.0,
            percentile_high=95.0,
            n_centroids=coordinator.n_centroids,
        )
        logger.info(f"[Island {island_idx}] Built {n_centroids} centroids")

        # Sort by score and add top programs
        valid_programs.sort(key=lambda x: x["score"], reverse=True)
        top_programs = valid_programs[:coordinator.n_centroids]

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
        logger.info(
            f"[Island {island_idx}] Initialized: {n_accepted} accepted, "
            f"archive size: {pool.size()}, best: {best_score:.1f}"
        )

        # Switch extractor to evolution phase (only on last island)
        if island_idx == coordinator.n_islands - 1:
            extractor.set_phase('evolution')

        return self.total_cost

    async def _generate_variants(
        self,
        seed_code: str,
        seed_score: float,
        fn_name: str,
        extractor: BehaviorExtractor,
        island_idx: int,
    ) -> tuple[list[dict], list[np.ndarray]]:
        """Generate variants for a single island."""
        n_variants = self.config.init.n_variants_per_seed
        model = self.config.init.variant_model or "gpt-4o-mini"

        logger.info(
            f"[Island {island_idx}] Generating {n_variants} variants with {model}"
        )

        # Create seed program for prompt building
        seed_prog = Program(code=seed_code, metadata={"score": seed_score})
        seed_eval = EvaluationResult(
            program_id=seed_prog.id,
            scores={'score': seed_score},
            is_valid=True,
        )
        parent = ProgramWithScore(seed_prog, seed_eval)

        # Build prompts for variants
        prompts = []
        for _ in range(n_variants):
            builder = PromptBuilder()
            builder.add_section("Problem", self.config.problem_description, priority=10)
            builder.add_section(
                "Signature",
                f"```python\n{self.config.function_signature}\n```",
                priority=20
            )
            builder.add_parents([parent], priority=30)
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

        logger.info(
            f"[Island {island_idx}] Evaluating {len(candidates)} candidates"
        )

        # Parallel evaluation
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
        eval_results = await asyncio.gather(*eval_tasks)

        # Collect valid programs and behavior vectors
        valid_programs = []
        behavior_vectors = []

        # Set extractor to init phase for this island
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

        # Also add the seed itself
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

        logger.info(
            f"[Island {island_idx}] Valid programs: {len(valid_programs)}"
        )
        return valid_programs, behavior_vectors

    async def initialize_all_islands(
        self,
        coordinator: IslandCoordinator,
        base_seed: str,
        base_result: dict,
        extractor: BehaviorExtractor,
    ) -> float:
        """
        Initialize all islands with diverse seeds and variants.

        This is the main entry point for island initialization:
        1. Generate diverse seeds (one per island)
        2. For each island, generate variants and compute centroids
        3. Populate each island's archive

        Args:
            coordinator: The IslandCoordinator to initialize
            base_seed: Initial seed program
            base_result: Evaluation result of base seed
            extractor: Behavior extractor

        Returns:
            Total cost incurred during initialization
        """
        # Step 1: Generate diverse seeds for each island
        seeds = await self.generate_island_seeds(
            base_seed, base_result, coordinator.n_islands
        )

        # Step 2: Initialize each island with its seed
        for i, (seed_code, seed_result) in enumerate(seeds):
            await self.initialize_island(
                i, coordinator, seed_code, seed_result, extractor
            )

        logger.info(
            f"[IslandDiversifier] All {coordinator.n_islands} islands initialized, "
            f"total cost: ${self.total_cost:.3f}"
        )
        return self.total_cost
