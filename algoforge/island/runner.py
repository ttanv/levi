"""Island Pipeline Runner: orchestrates evolution across multiple islands."""

import asyncio
import json
import logging
import random
import types
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Optional, Callable


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

from ..config import AlgoforgeConfig, AlgoforgeResult, BudgetConfig
from ..core import Program, EvaluationResult
from ..behavior import BehaviorExtractor
from ..llm import PromptBuilder, ProgramWithScore, OutputMode, get_llm_client, LLMRetryExhaustedError
from ..utils import ResilientProcessPool, extract_code, extract_fn_name
from ..pipeline.state import PipelineState
from ..pipeline.consumer import _generate_meta_advice
from .coordinator import IslandCoordinator
from .diversifier import IslandDiversifier

logger = logging.getLogger(__name__)


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


class IslandPipelineRunner:
    """
    Orchestrates evolution across multiple islands.

    Similar to PipelineRunner but distributes work across islands
    with round-robin sampling and automatic migration.
    """

    def __init__(
        self,
        config: AlgoforgeConfig,
        coordinator: IslandCoordinator,
        executor: ResilientProcessPool,
        output_dir: Optional[str] = None,
        culling_checkpoints: Optional[list[float]] = None,
        n_seed_elites: int = 1,
    ):
        self.config = config
        self.coordinator = coordinator
        self.executor = executor
        self.archive_lock = asyncio.Lock()
        self.code_queue = asyncio.Queue()
        self.state = PipelineState(config.budget)
        self.stop_event = asyncio.Event()

        # Culling configuration
        self.culling_checkpoints = culling_checkpoints or [0.25, 0.50, 0.75]
        self.n_seed_elites = n_seed_elites
        self._completed_culling_checkpoints: set[float] = set()

        self.output_dir: Optional[Path] = None
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> AlgoforgeResult:
        """Run the island evolution pipeline."""
        n_islands = self.coordinator.n_islands
        logger.info(
            f"[IslandPipeline] Starting with {n_islands} islands, "
            f"{self.config.pipeline.n_llm_workers} LLM workers, "
            f"{self.config.pipeline.n_eval_processes} eval workers"
        )

        # Set initial best score from initialized archives
        _, initial_best = self.coordinator.get_global_best()
        self.state.best_score_so_far = initial_best

        island_indices = list(range(n_islands))

        producers = [
            asyncio.create_task(
                self._llm_producer(worker_id=i, island_cycle=cycle(island_indices))
            )
            for i in range(self.config.pipeline.n_llm_workers)
        ]

        consumers = [
            asyncio.create_task(self._eval_consumer(worker_id=i))
            for i in range(self.config.pipeline.n_eval_processes)
        ]

        status_task = asyncio.create_task(self._status_monitor())

        try:
            await self._wait_for_completion()
        finally:
            self.stop_event.set()

            for task in producers:
                task.cancel()
            await asyncio.gather(*producers, return_exceptions=True)

            while not self.code_queue.empty():
                await asyncio.sleep(0.5)

            for task in consumers:
                task.cancel()
            await asyncio.gather(*consumers, return_exceptions=True)

            status_task.cancel()
            try:
                await status_task
            except asyncio.CancelledError:
                pass

            if self.output_dir:
                self._save_snapshot(final=True)

        return self._build_result()

    async def _llm_producer(self, worker_id: int, island_cycle: cycle) -> None:
        """Sample from islands in round-robin and generate code."""
        while not self.stop_event.is_set():
            if self.state.budget_exhausted:
                break

            try:
                island_idx = next(island_cycle)
                island = self.coordinator.get_island(island_idx)

                if island.pool.size() == 0:
                    await asyncio.sleep(0.1)
                    continue

                async with self.archive_lock:
                    sampler_name, model = island.pool.get_weighted_sampler_config()
                    n_parents = self.config.pipeline.n_parents + self.config.pipeline.n_inspirations
                    context = {"budget_progress": self.state.budget_progress}
                    sample = island.pool.sample(sampler_name, n_parents=n_parents, context=context)

                parent = sample.parent
                inspirations = [p for p in sample.inspirations if random.random() < 0.8]
                parents = [parent] + inspirations

                builder = PromptBuilder()
                builder.add_section("Problem", self.config.problem_description, priority=10)
                builder.add_section("Signature", f"```python\n{self.config.function_signature}\n```", priority=20)
                builder.add_parents([ProgramWithScore(p, None) for p in parents], priority=30)
                builder.set_output_mode(OutputMode.FULL)

                if self.state.current_meta_advice and random.random() < 0.8:
                    builder.add_section("Meta-Advice", self.state.current_meta_advice, priority=100)

                prompt = builder.build()

                self.state.llm_in_flight += 1
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
                    cost = response.cost
                except LLMRetryExhaustedError as e:
                    logger.warning(f"[LLM-{worker_id}] Error after retries: {e.last_error}")
                    continue
                except Exception as e:
                    logger.warning(f"[LLM-{worker_id}] Error: {e}")
                    await asyncio.sleep(1.0)
                    continue
                finally:
                    self.state.llm_in_flight -= 1

                self.state.add_cost(cost)

                if self.state.budget_exhausted:
                    self.stop_event.set()
                    break

                code = extract_code(content)
                if not code:
                    continue

                await self.code_queue.put({
                    "code": code,
                    "sampler": sampler_name,
                    "source_cell": sample.metadata.get("source_cell"),
                    "model": model,
                    "island_idx": island_idx,
                })

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[LLM-{worker_id}] Unexpected error: {e}")
                await asyncio.sleep(1.0)

    async def _eval_consumer(self, worker_id: int) -> None:
        """Evaluate code and add to source island."""
        fn_name = extract_fn_name(self.config.function_signature)

        while not self.stop_event.is_set() or not self.code_queue.empty():
            try:
                item = await asyncio.wait_for(self.code_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            island_idx = item["island_idx"]

            self.state.eval_in_flight += 1
            try:
                result = await self.executor.run(
                    _evaluate_code,
                    item["code"],
                    self.config.score_fn,
                    self.config.inputs,
                    fn_name,
                    timeout=self.config.pipeline.eval_timeout
                )
            except TimeoutError:
                result = {"error": "Timeout"}
            except Exception as e:
                result = {"error": str(e)}
            finally:
                self.state.eval_in_flight -= 1

            async with self.archive_lock:
                if "error" not in result:
                    program = Program(code=item["code"])
                    eval_result = EvaluationResult(
                        scores=result,
                        is_valid=True,
                    )

                    accepted = self.coordinator.add_to_island(island_idx, program, eval_result)
                    self.coordinator.process_all_incoming()

                    island = self.coordinator.get_island(island_idx)
                    island.pool.update_sampler(item["sampler"], item["source_cell"], success=accepted)

                    if accepted:
                        self.state.record_accept()
                    else:
                        self.state.record_reject()

                    score = result.get('score', 0)
                    is_new_best = score > self.state.best_score_so_far

                    self.state.record_score(
                        score=score,
                        accepted=accepted,
                        sampler=item["sampler"],
                        archive_size=self.coordinator.get_total_archive_size(),
                    )

                    if is_new_best:
                        status = "NEW BEST ★"
                    elif accepted:
                        status = "accepted"
                    else:
                        status = "rejected"

                    logger.info(
                        f"[Eval #{self.state.eval_count}] Island {island_idx} {item['sampler']:15s} "
                        f"{status:12s} | score: {score:.1f} | best: {self.state.best_score_so_far:.1f} | "
                        f"cost: ${self.state.total_cost:.3f}"
                    )
                else:
                    island = self.coordinator.get_island(island_idx)
                    island.pool.update_sampler(item["sampler"], item["source_cell"], success=False)
                    self.state.record_error(result["error"])
                    logger.info(
                        f"[Eval #{self.state.eval_count}] Island {island_idx} ERROR: {result['error'][:50]}"
                    )

            # Generate meta-advice periodically to reduce errors
            if (self.config.meta_advice.enabled and
                self.state.should_generate_meta_advice(self.config.meta_advice.interval)):
                asyncio.create_task(_generate_meta_advice(self.config, self.state))

            # Check for culling checkpoints
            self._check_culling()

            if self.output_dir and self.state.eval_count % 10 == 0:
                try:
                    self._save_snapshot()
                except Exception as e:
                    logger.warning(f"[Snapshot] Failed to save: {e}")

    async def _wait_for_completion(self) -> None:
        while not self.state.budget_exhausted:
            await asyncio.sleep(1.0)

    def _check_culling(self) -> None:
        """Check if we've hit a culling checkpoint and perform culling if so."""
        progress = self.state.budget_progress

        for checkpoint in self.culling_checkpoints:
            if checkpoint in self._completed_culling_checkpoints:
                continue
            if progress >= checkpoint:
                logger.info(f"[Culling] Reached {checkpoint*100:.0f}% budget checkpoint")
                result = self.coordinator.perform_culling(n_seed_elites=self.n_seed_elites)
                self._completed_culling_checkpoints.add(checkpoint)
                logger.info(
                    f"[Culling] Complete: culled {result['culled']} islands, "
                    f"cleared {result.get('cleared_elites', 0)} elites, "
                    f"seeded {result.get('seeded_elites', 0)} elites"
                )

    async def _status_monitor(self) -> None:
        while not self.stop_event.is_set():
            await asyncio.sleep(30.0)
            stats = self.coordinator.get_stats()
            logger.info(
                f"[Status] Cost: ${self.state.total_cost:.3f} | "
                f"Evals: {self.state.eval_count} | "
                f"LLM: {self.state.llm_in_flight} | "
                f"Eval: {self.state.eval_in_flight} | "
                f"Archives: {stats['total_accepted']} | "
                f"Migrations: {stats['total_migrations']} | "
                f"Best: {self.state.best_score_so_far:.1f}"
            )

    def _save_snapshot(self, final: bool = False) -> None:
        """Save snapshot of all islands."""
        if not self.output_dir:
            return

        stats = self.coordinator.get_stats()

        island_elites = []
        for island in self.coordinator.islands:
            elites = []
            for cell_idx, elite in island.pool.get_elites().items():
                elites.append({
                    "cell_index": int(cell_idx),
                    "score": float(elite.result.primary_score),
                    "code": elite.program.code,
                })
            elites.sort(key=lambda x: x["score"], reverse=True)
            island_elites.append({
                "index": island.index,
                "archive_size": island.pool.size(),
                "best_score": island.best_score,
                "eval_count": island.eval_count,
                "top_elites": elites[:5],
            })

        global_best, global_best_score = self.coordinator.get_global_best()

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "run_state": {
                "elapsed_seconds": self.state.elapsed_seconds,
                "total_cost": self.state.total_cost,
                "eval_count": self.state.eval_count,
                "accept_count": self.state.accept_count,
                "error_count": self.state.error_count,
                "best_score": self.state.best_score_so_far,
            },
            "coordinator_stats": stats,
            "islands": island_elites,
            "global_best": {
                "score": global_best_score,
                "code": global_best.code if global_best else None,
            },
        }

        filepath = self.output_dir / "snapshot.json"
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)

        if final:
            logger.info(f"[Snapshot] Saved final snapshot to {filepath}")

    def _build_result(self) -> AlgoforgeResult:
        global_best, global_best_score = self.coordinator.get_global_best()
        return AlgoforgeResult(
            best_program=global_best.code if global_best else "",
            best_score=global_best_score,
            total_evaluations=self.state.eval_count,
            total_cost=self.state.total_cost,
            archive_size=self.coordinator.get_total_archive_size(),
            runtime_seconds=self.state.elapsed_seconds,
            score_history=self.state.get_score_history_list(),
        )


async def run_islands_async(
    config: AlgoforgeConfig,
    n_islands: int = 5,
    culling_checkpoints: Optional[list[float]] = None,
    n_seed_elites: int = 1,
    migration_interval: Optional[int] = None,
) -> AlgoforgeResult:
    """
    Run island-based evolution with periodic culling.

    Args:
        config: AlgoForge configuration
        n_islands: Number of islands to run
        culling_checkpoints: Budget fractions at which to cull bottom half of islands
                            (default: [0.25, 0.50, 0.75])
        n_seed_elites: Number of top elites to seed into culled islands
        migration_interval: Evals per island before migration triggers (default: 100).
                           Set to a very large number to disable migration.

    This is the async entry point for island evolution.
    """
    _setup_logging()
    if culling_checkpoints is None:
        culling_checkpoints = [0.25, 0.50, 0.75]

    logger.info(f"[Islands] Starting with {n_islands} islands")
    logger.info(f"[Islands] Culling at: {[f'{c*100:.0f}%' for c in culling_checkpoints]}")

    extractor = BehaviorExtractor(
        ast_features=config.behavior.ast_features,
        score_keys=config.behavior.score_keys,
        init_noise=config.behavior.init_noise,
    )

    coordinator_kwargs = {
        "n_islands": n_islands,
        "behavior_extractor": extractor,
        "n_centroids": config.cvt.n_centroids,
    }
    if migration_interval is not None:
        coordinator_kwargs["migration_interval"] = migration_interval

    coordinator = IslandCoordinator(**coordinator_kwargs)

    for island in coordinator.islands:
        for pair in config.sampler_model_pairs:
            island.pool.register_sampler_model_pair(
                pair.sampler, pair.model, pair.weight, pair.temperature, pair.n_cycles
            )

    executor = ResilientProcessPool(max_workers=config.pipeline.n_eval_processes)

    try:
        fn_name = extract_fn_name(config.function_signature)
        logger.info("[Islands] Evaluating seed program")

        seed_result = await executor.run(
            _evaluate_code,
            config.seed_program,
            config.score_fn,
            config.inputs,
            fn_name,
            timeout=config.pipeline.eval_timeout
        )

        if "error" in seed_result:
            raise RuntimeError(f"Seed evaluation error: {seed_result['error']}")

        seed_score = seed_result.get('score', 0.0)
        logger.info(f"[Islands] Seed score: {seed_score:.1f}")

        if config.init.enabled:
            logger.info("[Islands] Initializing islands with diverse seeds")
            diversifier = IslandDiversifier(config, executor)
            init_cost = await diversifier.initialize_all_islands(
                coordinator, config.seed_program, seed_result, extractor
            )
            logger.info(f"[Islands] Init complete, cost: ${init_cost:.3f}")
        else:
            raise RuntimeError("Island mode requires init.enabled=True")

        runner = IslandPipelineRunner(
            config,
            coordinator,
            executor,
            output_dir=config.output_dir,
            culling_checkpoints=culling_checkpoints,
            n_seed_elites=n_seed_elites,
        )
        runner.state.total_cost = init_cost if config.init.enabled else 0.0

        result = await runner.run()

        logger.info(f"[Islands] Complete - best score: {result.best_score:.1f}, "
                    f"evals: {result.total_evaluations}, cost: ${result.total_cost:.3f}, "
                    f"archive: {result.archive_size}")
        if config.output_dir:
            logger.info(f"[Islands] Snapshot: {config.output_dir}/snapshot.json")
        code_preview = result.best_program[:500]
        if len(result.best_program) > 500:
            code_preview += "..."
        logger.info(f"[Islands] Best program:\n{code_preview}")

        return result

    finally:
        executor.shutdown()


def run_islands(
    config: AlgoforgeConfig,
    n_islands: int = 5,
    culling_checkpoints: Optional[list[float]] = None,
    n_seed_elites: int = 1,
    migration_interval: Optional[int] = None,
) -> AlgoforgeResult:
    """
    Run island-based evolution with periodic culling.

    Synchronous entry point that handles async internally.
    """
    return asyncio.run(run_islands_async(
        config,
        n_islands=n_islands,
        culling_checkpoints=culling_checkpoints,
        n_seed_elites=n_seed_elites,
        migration_interval=migration_interval,
    ))
