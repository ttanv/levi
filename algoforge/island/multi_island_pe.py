"""
Multi-Island Punctuated Equilibrium Runner.

Quick-and-dirty implementation of multi-island evolution where:
- 4 islands share the same centroids.json behavior map
- Each island seeded with 1 unique LLM-generated seed (no variants)
- Evolution happens independently on all islands
- Every N evals: cross-island PE compares best from each island,
  accepts only if it beats the weakest island's best
"""

import asyncio
import json
import logging
import random
import types
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from ..config import AlgoforgeConfig, AlgoforgeResult
from ..core import Program, EvaluationResult
from ..pool import CVTMAPElitesPool
from ..behavior import BehaviorExtractor, FeatureVector
from ..llm import PromptBuilder, ProgramWithScore, OutputMode, get_llm_client, set_llm_client, clear_llm_client
from ..llm.unified_client import UnifiedLLMClient, UnifiedLLMClientConfig
from ..utils import ResilientProcessPool, extract_code, extract_fn_name
from ..pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging for island runner."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("litellm").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


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
        result = score_fn(fn, inputs)
        return result
    except MemoryError:
        return {"error": "MemoryError during scoring"}
    except Exception as e:
        return {"error": f"Scoring error: {e}"}


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


CROSS_ISLAND_PE_PROMPT = """# Cross-Island Paradigm Shift Challenge

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Current Best Solutions from Each Island

{n_islands} islands have been evolving independently. Below are the best solutions from each:

{island_solutions}

## Your Challenge: PARADIGM SHIFT

Analyze ALL island solutions and synthesize a **fundamentally new approach** that:
1. Combines insights from multiple islands
2. Explores untapped regions of the solution space
3. Uses a different algorithmic paradigm than any individual island

### Analysis Steps:
1. **Identify each island's paradigm**: What strategy does each island use?
2. **Find synergies**: What could be combined from different islands?
3. **Find gaps**: What paradigms are NOT represented across all islands?

### Critical Requirements:
- Your function signature MUST match exactly: `{function_signature}`
- Use only standard Python libraries (numpy, collections, itertools, math, heapq, functools, etc.)
- The code must be syntactically valid and complete
- Include ALL necessary imports at the top
- Do NOT use placeholders or ellipses

## Output
Output ONLY complete, runnable Python code in a ```python block.
"""


class MultiIslandPERunner:
    """
    Multi-Island PE Runner with shared centroids and cross-island paradigm shifts.

    Key features:
    - All islands share the same centroid space from centroids.json
    - Each island gets 1 unique seed (no variants)
    - Round-robin evolution across islands
    - Cross-island PE every N evals comparing all island bests
    """

    def __init__(
        self,
        config: AlgoforgeConfig,
        centroids_file: str,
        n_islands: int = 4,
        pe_interval: int = 5,
    ):
        self.config = config
        self.centroids_file = Path(centroids_file)
        self.n_islands = n_islands
        self.pe_interval = pe_interval
        self.fn_name = extract_fn_name(config.function_signature)

        # Load shared centroids
        self.centroids_data = self._load_centroids()
        self.bounds = self.centroids_data["bounds"]
        self.centroid_vectors = self._build_centroid_vectors()

        # Create behavior extractor with deterministic bounds
        self.extractor = BehaviorExtractor(
            ast_features=config.behavior.ast_features,
            score_keys=config.behavior.score_keys,
            init_noise=0.0,
        )
        # Set fixed bounds from centroids.json (convert list to tuple)
        bounds_tuple = {k: tuple(v) for k, v in self.bounds.items()}
        self.extractor.set_fixed_bounds(bounds_tuple)

        # Create islands with shared centroids
        self.islands: list[CVTMAPElitesPool] = []
        self._create_islands()

        # State tracking
        self.total_cost = 0.0
        self.eval_count = 0
        self.pe_events = []

    def _load_centroids(self) -> dict:
        """Load centroids from JSON file."""
        with open(self.centroids_file) as f:
            data = json.load(f)
        logger.info(f"[MultiIslandPE] Loaded {len(data['centroids'])} centroids from {self.centroids_file}")
        return data

    def _build_centroid_vectors(self) -> np.ndarray:
        """Build centroid vectors in normalized [0,1] space."""
        features = self.config.behavior.ast_features
        centroids = []

        for c in self.centroids_data["centroids"]:
            vec = []
            for f in features:
                raw_val = c["vector"][f]
                low, high = self.bounds[f]
                # Normalize to [0,1]
                if high > low:
                    normalized = (raw_val - low) / (high - low)
                else:
                    normalized = 0.5
                vec.append(np.clip(normalized, 0, 1))
            centroids.append(vec)

        return np.array(centroids)

    def _create_islands(self) -> None:
        """Create N islands with identical centroid configuration."""
        for i in range(self.n_islands):
            pool = CVTMAPElitesPool(
                behavior_extractor=self.extractor,
                n_centroids=len(self.centroid_vectors),
                defer_centroids=True,
            )
            # Set the shared centroids directly
            pool._centroids = self.centroid_vectors.copy()
            pool._n_centroids = len(self.centroid_vectors)
            pool._mins = np.zeros(len(self.config.behavior.ast_features))
            pool._maxs = np.ones(len(self.config.behavior.ast_features))
            pool._ranges = np.ones(len(self.config.behavior.ast_features))

            # Register sampler-model pairs
            for pair in self.config.sampler_model_pairs:
                pool.register_sampler_model_pair(
                    pair.sampler, pair.model, pair.weight, pair.temperature, pair.n_cycles
                )

            self.islands.append(pool)

        logger.info(f"[MultiIslandPE] Created {self.n_islands} islands with {len(self.centroid_vectors)} shared centroids")

    async def generate_diverse_seeds(
        self,
        executor: ResilientProcessPool,
    ) -> list[tuple[str, float]]:
        """
        Generate N unique seeds via LLM (one per island, no variants).

        Uses the config.seed_program as seed #1, then generates seeds #2-N
        with each shown previous seeds for diversity.
        """
        model = self.config.init.diversity_model or "gpt-4"

        # Start with the base seed
        seed_result = await executor.run(
            _evaluate_code,
            self.config.seed_program,
            self.config.score_fn,
            self.config.inputs,
            self.fn_name,
            timeout=self.config.pipeline.eval_timeout
        )

        if "error" in seed_result:
            raise RuntimeError(f"Seed evaluation error: {seed_result['error']}")

        seed_score = seed_result.get('score', 0.0)
        logger.info(f"[MultiIslandPE] Base seed score: {seed_score:.1f}")

        seeds = [(self.config.seed_program, seed_score)]

        # Generate remaining seeds
        for i in range(self.n_islands - 1):
            existing_seeds_text = "\n\n---\n\n".join([
                f"### Seed {j+1} (Score: {score:.1f}):\n```python\n{code}\n```"
                for j, (code, score) in enumerate(seeds)
            ])

            prompt = DIVERSITY_SEED_PROMPT.format(
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
                    logger.warning(f"  [Seed {i+2}] Failed to extract code")
                    # Use base seed as fallback
                    seeds.append((self.config.seed_program, seed_score))
                    continue

                result = await executor.run(
                    _evaluate_code,
                    new_code,
                    self.config.score_fn,
                    self.config.inputs,
                    self.fn_name,
                    timeout=self.config.pipeline.eval_timeout
                )

                if "error" not in result:
                    new_score = result.get('score', 0.0)
                    seeds.append((new_code, new_score))
                    logger.info(f"  [Seed {i+2}] OK - score: {new_score:.1f}")
                else:
                    logger.warning(f"  [Seed {i+2}] Failed: {result['error'][:80]}")
                    seeds.append((self.config.seed_program, seed_score))

            except Exception as e:
                logger.warning(f"  [Seed {i+2}] Error: {e}")
                seeds.append((self.config.seed_program, seed_score))

        logger.info(f"[MultiIslandPE] Generated {len(seeds)} seeds")
        return seeds

    def seed_islands(self, seeds: list[tuple[str, float]]) -> None:
        """Seed each island with one unique seed."""
        for i, (code, score) in enumerate(seeds[:self.n_islands]):
            program = Program(code=code, metadata={"seed": True})
            # Create evaluation result - we need scores dict
            eval_result = EvaluationResult(
                scores={"score": score},
                is_valid=True,
            )

            accepted, cell_idx = self.islands[i].add(program, eval_result)
            logger.info(f"[MultiIslandPE] Island {i} seeded: score={score:.1f}, cell={cell_idx}, accepted={accepted}")

    def get_island_best_elites(self) -> list[tuple[int, float, str]]:
        """
        Returns (island_idx, score, code) for each island's best elite.
        """
        results = []
        for i, pool in enumerate(self.islands):
            if pool.size() > 0:
                best_prog = pool.best()
                best_score = pool._best_score
                results.append((i, best_score, best_prog.code))
            else:
                results.append((i, float('-inf'), ""))
        return results

    async def trigger_cross_island_pe(self, executor: ResilientProcessPool) -> dict:
        """
        Cross-island paradigm shift event.

        1. Get best elite from each of N islands
        2. Build paradigm shift prompt with all N approaches
        3. Generate new solution with heavy model
        4. Accept ONLY if score > weakest island's best score
        5. If accepted, add to weakest island's archive
        """
        stats = {
            "triggered": True,
            "paradigm_score": None,
            "paradigm_accepted": False,
            "target_island": None,
            "island_scores": {},
        }

        # Get best from each island
        island_bests = self.get_island_best_elites()

        # Record island scores
        for idx, score, _ in island_bests:
            stats["island_scores"][idx] = score

        # Sort by score to find weakest
        sorted_islands = sorted(island_bests, key=lambda x: x[1])
        weakest_idx, weakest_score, _ = sorted_islands[0]

        logger.info(f"[PE] Island scores: {[(i, f'{s:.1f}') for i, s, _ in island_bests]}")
        logger.info(f"[PE] Weakest island: {weakest_idx} (score: {weakest_score:.1f})")

        # Build prompt with all island solutions
        island_solutions = []
        for i, score, code in island_bests:
            if code:
                island_solutions.append(
                    f"### Island {i} (Score: {score:.1f})\n```python\n{code}\n```"
                )

        if len(island_solutions) < 2:
            logger.warning("[PE] Not enough island solutions for PE")
            stats["triggered"] = False
            return stats

        prompt = CROSS_ISLAND_PE_PROMPT.format(
            problem_description=self.config.problem_description,
            function_signature=self.config.function_signature,
            n_islands=self.n_islands,
            island_solutions="\n\n".join(island_solutions),
        )

        # Generate paradigm shift
        heavy_model = self.config.punctuated_equilibrium.heavy_model
        if not heavy_model:
            heavy_model = self.config.sampler_model_pairs[0].model

        try:
            llm = get_llm_client()

            extras = {}
            if self.config.punctuated_equilibrium.reasoning_effort:
                if self.config.punctuated_equilibrium.reasoning_effort == "disabled":
                    extras["extra_body"] = {"reasoning": {"enabled": False}}
                else:
                    extras["reasoning_effort"] = self.config.punctuated_equilibrium.reasoning_effort

            response = await llm.acompletion(
                model=heavy_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.punctuated_equilibrium.temperature,
                max_tokens=4096,
                timeout=300,
                **extras,
            )
            content = response.content
            self.total_cost += response.cost

        except Exception as e:
            logger.warning(f"[PE] Paradigm shift generation failed: {e}")
            return stats

        paradigm_code = extract_code(content)
        if not paradigm_code:
            logger.warning("[PE] Failed to extract paradigm shift code")
            return stats

        # Print paradigm shift code
        print("\n" + "="*60)
        print("[PE] CROSS-ISLAND PARADIGM SHIFT CODE:")
        print("="*60)
        print(paradigm_code)
        print("="*60 + "\n")

        # Evaluate
        try:
            result = await executor.run(
                _evaluate_code,
                paradigm_code,
                self.config.score_fn,
                self.config.inputs,
                self.fn_name,
                timeout=self.config.pipeline.eval_timeout
            )
        except Exception as e:
            logger.warning(f"[PE] Paradigm shift evaluation failed: {e}")
            return stats

        if "error" in result:
            logger.info(f"[PE] Paradigm shift eval error: {result['error'][:100]}")
            return stats

        paradigm_score = result.get("score", 0.0)
        stats["paradigm_score"] = paradigm_score

        # Accept only if beats weakest island
        if paradigm_score > weakest_score:
            program = Program(code=paradigm_code, metadata={
                "source": "cross_island_pe",
                "pe_type": "paradigm_shift",
            })
            eval_result = EvaluationResult(
                scores=result,
                is_valid=True,
            )

            accepted, cell_idx = self.islands[weakest_idx].add(program, eval_result)
            stats["paradigm_accepted"] = accepted
            stats["target_island"] = weakest_idx

            logger.info(
                f"[PE] ACCEPTED: score={paradigm_score:.1f} > weakest={weakest_score:.1f}, "
                f"added to island {weakest_idx}, cell={cell_idx}"
            )
        else:
            logger.info(
                f"[PE] REJECTED: score={paradigm_score:.1f} <= weakest={weakest_score:.1f}"
            )

        self.pe_events.append(stats)
        return stats

    async def run_evolution_step(
        self,
        island_idx: int,
        executor: ResilientProcessPool,
    ) -> dict:
        """Run single evolution step on one island."""
        pool = self.islands[island_idx]

        if pool.size() == 0:
            return {"error": "Island empty"}

        # Sample parent
        context = {"budget_progress": 0.5}  # Simplified
        sampler_name, model = pool.get_weighted_sampler_config()
        n_parents = self.config.pipeline.n_parents + self.config.pipeline.n_inspirations
        sample = pool.sample(sampler_name, n_parents=n_parents, context=context)

        parent = sample.parent
        inspirations = [p for p in sample.inspirations if random.random() < 0.8]
        parents = [parent] + inspirations

        # Build prompt
        builder = PromptBuilder()
        builder.add_section("Problem", self.config.problem_description, priority=10)
        builder.add_section("Signature", f"```python\n{self.config.function_signature}\n```", priority=20)
        builder.add_parents([ProgramWithScore(p, None) for p in parents], priority=30)
        builder.set_output_mode(OutputMode.FULL)
        prompt = builder.build()

        # Generate
        try:
            llm = get_llm_client()
            response = await llm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.pipeline.temperature,
                max_tokens=self.config.pipeline.max_tokens,
                timeout=300,
            )
            content = response.content
            self.total_cost += response.cost
        except Exception as e:
            logger.warning(f"[Evo-{island_idx}] LLM error: {e}")
            return {"error": str(e)}

        code = extract_code(content)
        if not code:
            return {"error": "No code extracted"}

        # Evaluate
        try:
            result = await executor.run(
                _evaluate_code,
                code,
                self.config.score_fn,
                self.config.inputs,
                self.fn_name,
                timeout=self.config.pipeline.eval_timeout
            )
        except Exception as e:
            return {"error": str(e)}

        if "error" in result:
            pool.update_sampler(sampler_name, sample.metadata.get("source_cell"), success=False)
            return {"error": result["error"]}

        # Add to archive
        program = Program(code=code)
        eval_result = EvaluationResult(
            scores=result,
            is_valid=True,
        )

        accepted, cell_idx = pool.add(program, eval_result)
        pool.update_sampler(sampler_name, sample.metadata.get("source_cell"), success=accepted)

        score = result.get('score', 0)
        self.eval_count += 1

        status = "accepted" if accepted else "rejected"
        logger.info(
            f"[Eval #{self.eval_count}] Island {island_idx} {sampler_name:15s} {status:12s} | "
            f"score: {score:.1f} | best: {pool._best_score:.1f} | cost: ${self.total_cost:.3f}"
        )

        return {
            "accepted": accepted,
            "score": score,
            "cell": cell_idx,
            "island": island_idx,
        }

    async def run_async(self, max_evals: int = 100) -> AlgoforgeResult:
        """
        Main evolution loop.

        1. Generate and seed islands with diverse seeds
        2. Round-robin evolution across islands
        3. Trigger cross-island PE every pe_interval evals
        """
        # Initialize unified LLM client
        llm_config = UnifiedLLMClientConfig(
            local_endpoints=self.config.llm.local_endpoints,
            model_info=self.config.llm.model_info,
            max_retries=self.config.llm.max_retries,
            retry_delay=self.config.llm.retry_delay,
            retry_backoff=self.config.llm.retry_backoff,
            default_temperature=self.config.pipeline.temperature,
            default_max_tokens=self.config.pipeline.max_tokens,
            batch_size=self.config.llm.batch_size,
            batch_max_wait_ms=self.config.llm.batch_max_wait_ms,
        )
        llm_client = UnifiedLLMClient(llm_config)
        set_llm_client(llm_client)

        executor = ResilientProcessPool(max_workers=self.config.pipeline.n_eval_processes)

        try:
            # Phase 1: Generate diverse seeds
            logger.info(f"[MultiIslandPE] Phase 1: Generating {self.n_islands} diverse seeds")
            seeds = await self.generate_diverse_seeds(executor)

            # Phase 2: Seed islands
            logger.info("[MultiIslandPE] Phase 2: Seeding islands")
            self.seed_islands(seeds)

            # Phase 3: Evolution with cross-island PE
            logger.info(f"[MultiIslandPE] Phase 3: Evolution (max_evals={max_evals}, pe_interval={self.pe_interval})")

            island_cycle = 0
            while self.eval_count < max_evals:
                # Check for PE event
                if self.eval_count > 0 and self.eval_count % self.pe_interval == 0:
                    logger.info(f"\n[MultiIslandPE] === Cross-Island PE Event @ eval {self.eval_count} ===")
                    await self.trigger_cross_island_pe(executor)

                # Round-robin evolution step
                island_idx = island_cycle % self.n_islands
                await self.run_evolution_step(island_idx, executor)
                island_cycle += 1

                # Check budget
                if self.total_cost >= self.config.budget.dollars:
                    logger.info(f"[MultiIslandPE] Budget exhausted: ${self.total_cost:.3f}")
                    break

            # Build result
            best_code = ""
            best_score = float('-inf')
            total_archive = 0

            for i, pool in enumerate(self.islands):
                total_archive += pool.size()
                if pool.size() > 0 and pool._best_score > best_score:
                    best_score = pool._best_score
                    best_code = pool.best().code

            # Final stats
            logger.info("\n" + "="*60)
            logger.info("[MultiIslandPE] FINAL RESULTS")
            logger.info("="*60)
            for i, pool in enumerate(self.islands):
                logger.info(f"  Island {i}: archive={pool.size()}, best={pool._best_score:.1f}")
            logger.info(f"  Global best: {best_score:.1f}")
            logger.info(f"  Total evals: {self.eval_count}")
            logger.info(f"  Total cost: ${self.total_cost:.3f}")
            logger.info(f"  PE events: {len(self.pe_events)}")
            logger.info("="*60)

            return AlgoforgeResult(
                best_program=best_code,
                best_score=best_score,
                total_evaluations=self.eval_count,
                total_cost=self.total_cost,
                archive_size=total_archive,
                runtime_seconds=0,  # Not tracked here
                score_history=[],
            )

        finally:
            executor.shutdown()
            clear_llm_client()

    def run(self, max_evals: int = 100) -> AlgoforgeResult:
        """Synchronous wrapper for run_async."""
        return asyncio.run(self.run_async(max_evals))


async def run_multi_island_pe_async(
    config: AlgoforgeConfig,
    centroids_file: str,
    n_islands: int = 4,
    pe_interval: int = 5,
    max_evals: int = 100,
) -> AlgoforgeResult:
    """
    Run multi-island PE evolution.

    Args:
        config: AlgoForge configuration
        centroids_file: Path to centroids.json with shared behavior space
        n_islands: Number of islands (default 4)
        pe_interval: Evals between cross-island PE events (default 5)
        max_evals: Maximum evaluations (default 100)

    Returns:
        AlgoforgeResult with best program found
    """
    _setup_logging()

    runner = MultiIslandPERunner(
        config=config,
        centroids_file=centroids_file,
        n_islands=n_islands,
        pe_interval=pe_interval,
    )

    return await runner.run_async(max_evals=max_evals)


def run_multi_island_pe(
    config: AlgoforgeConfig,
    centroids_file: str,
    n_islands: int = 4,
    pe_interval: int = 5,
    max_evals: int = 100,
) -> AlgoforgeResult:
    """
    Run multi-island PE evolution (synchronous).

    Args:
        config: AlgoForge configuration
        centroids_file: Path to centroids.json with shared behavior space
        n_islands: Number of islands (default 4)
        pe_interval: Evals between cross-island PE events (default 5)
        max_evals: Maximum evaluations (default 100)

    Returns:
        AlgoforgeResult with best program found
    """
    return asyncio.run(run_multi_island_pe_async(
        config=config,
        centroids_file=centroids_file,
        n_islands=n_islands,
        pe_interval=pe_interval,
        max_evals=max_evals,
    ))
