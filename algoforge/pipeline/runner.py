"""Pipeline orchestration: coordinates producers and consumers."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import AlgoforgeConfig, AlgoforgeResult
from ..equilibrium import PunctuatedEquilibrium
from ..pool import CVTMAPElitesPool
from ..utils import ResilientProcessPool
from .state import PipelineState
from .producer import llm_producer
from .consumer import eval_consumer

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(
        self,
        config: AlgoforgeConfig,
        pool: CVTMAPElitesPool,
        executor: ResilientProcessPool,
        output_dir: Optional[str] = None,
        init_cost: float = 0.0,
    ):
        self.config = config
        self.pool = pool
        self.executor = executor
        self.archive_lock = asyncio.Lock()
        self.code_queue = asyncio.Queue()
        self.state = PipelineState(config.budget)
        self.state.total_cost = init_cost  # Include init phase cost
        self.state.best_score_so_far = pool.get_stats().get("best_score", float('-inf'))
        self.stop_event = asyncio.Event()

        # Snapshot output directory
        self.output_dir: Optional[Path] = None
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Punctuated Equilibrium if enabled
        self.pe: Optional[PunctuatedEquilibrium] = None
        if config.punctuated_equilibrium.enabled:
            self.pe = PunctuatedEquilibrium(
                config=config,
                pool=pool,
                executor=executor,
                archive_lock=self.archive_lock,
            )

    async def run(self) -> AlgoforgeResult:
        logger.info(f"[Pipeline] Starting with {self.config.pipeline.n_llm_workers} LLM workers, "
                    f"{self.config.pipeline.n_eval_processes} eval workers")

        producers = [
            asyncio.create_task(
                llm_producer(
                    worker_id=i,
                    code_queue=self.code_queue,
                    pool=self.pool,
                    archive_lock=self.archive_lock,
                    config=self.config,
                    state=self.state,
                    stop_event=self.stop_event,
                )
            )
            for i in range(self.config.pipeline.n_llm_workers)
        ]

        consumers = [
            asyncio.create_task(
                eval_consumer(
                    worker_id=i,
                    code_queue=self.code_queue,
                    pool=self.pool,
                    archive_lock=self.archive_lock,
                    executor=self.executor,
                    config=self.config,
                    state=self.state,
                    stop_event=self.stop_event,
                    snapshot_callback=self.save_snapshot if self.output_dir else None,
                )
            )
            for i in range(self.config.pipeline.n_eval_processes)
        ]

        status_task = asyncio.create_task(self._status_monitor())

        # Launch Punctuated Equilibrium monitor if enabled
        pe_task = None
        if self.pe:
            pe_task = asyncio.create_task(self._pe_monitor())
            logger.info(f"[Pipeline] Punctuated equilibrium enabled (interval={self.config.punctuated_equilibrium.interval})")

        try:
            await self._wait_for_completion()
        finally:
            self.stop_event.set()

            for task in producers:
                task.cancel()
            await asyncio.gather(*producers, return_exceptions=True)

            # Wait for queue to drain
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

            # Cancel PE task if running
            if pe_task:
                pe_task.cancel()
                try:
                    await pe_task
                except asyncio.CancelledError:
                    pass

            # Final snapshot on completion
            if self.output_dir:
                self.save_snapshot(final=True)

        return self._build_result()

    async def _wait_for_completion(self) -> None:
        while not self.state.budget_exhausted:
            await asyncio.sleep(1.0)

    async def _status_monitor(self) -> None:
        while not self.stop_event.is_set():
            try:
                await asyncio.sleep(30.0)
                best_score = self.pool.get_stats().get("best_score", 0.0)
                logger.info(
                    f"[Status] Cost: ${self.state.total_cost:.3f} | "
                    f"Evals: {self.state.eval_count} | "
                    f"LLM in-flight: {self.state.llm_in_flight} | "
                    f"Eval in-flight: {self.state.eval_in_flight} | "
                    f"Archive: {self.pool.size()} | "
                    f"Best: {best_score:.1f}"
                )
            except asyncio.CancelledError:
                raise  # Let cancellation propagate
            except Exception as e:
                logger.error(f"[Status] Monitor error (will retry): {e}")
                await asyncio.sleep(5.0)  # Brief pause before retry

    async def _pe_monitor(self) -> None:
        """Monitor for punctuated equilibrium trigger conditions."""
        pe_config = self.config.punctuated_equilibrium

        while not self.stop_event.is_set():
            try:
                await asyncio.sleep(2.0)  # Check every 2 seconds

                if self.state.budget_exhausted:
                    break

                # Check if we should trigger PE
                if (self.state.eval_count > 0 and
                    self.state.eval_count % pe_config.interval == 0 and
                    self.state.eval_count != self.state.last_pe_eval_count):

                    self.state.last_pe_eval_count = self.state.eval_count
                    self.state.pe_trigger_count += 1

                    logger.info(f"[PE] Triggering punctuated equilibrium #{self.state.pe_trigger_count} "
                               f"at eval {self.state.eval_count}")

                    try:
                        stats = await self.pe.trigger(self.state.eval_count, self.state.budget_progress)
                        # Add PE cost to total
                        self.state.add_cost(stats["total_cost"])
                    except Exception as e:
                        logger.error(f"[PE] Trigger failed (will continue): {e}")

            except asyncio.CancelledError:
                raise  # Let cancellation propagate
            except Exception as e:
                logger.error(f"[PE] Monitor error (will retry): {e}")
                await asyncio.sleep(5.0)  # Brief pause before retry

    def save_snapshot(self, final: bool = False) -> None:
        """Save current archive state to JSON file (overwrites previous)."""
        if not self.output_dir:
            return

        # Get full archive with all programs
        snapshot = self.pool.get_archive_snapshot()

        # Add run state
        snapshot["run_state"] = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": self.state.elapsed_seconds,
            "total_cost": self.state.total_cost,
            "eval_count": self.state.eval_count,
            "accept_count": self.state.accept_count,
            "error_count": self.state.error_count,
            "best_score": self.state.best_score_so_far,
        }

        # Add score history
        snapshot["score_history"] = [
            {
                "eval_number": entry.eval_number,
                "score": entry.score,
                "best_score": entry.best_score,
                "timestamp": entry.timestamp,
                "accepted": entry.accepted,
                "sampler": entry.sampler,
                "archive_size": entry.archive_size,
                "cell_index": entry.cell_index,
                "is_punctuated_equilibrium": entry.is_punctuated_equilibrium,
            }
            for entry in self.state.score_history
        ]

        # Save to file (overwrite)
        filepath = self.output_dir / "snapshot.json"
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)

        if final:
            logger.info(f"[Snapshot] Saved final snapshot to {filepath}")

    def _build_result(self) -> AlgoforgeResult:
        best = self.pool.best()
        return AlgoforgeResult(
            best_program=best.code if best else "",
            best_score=self.pool.get_stats().get("best_score", 0.0),
            total_evaluations=self.state.eval_count,
            total_cost=self.state.total_cost,
            archive_size=self.pool.size(),
            runtime_seconds=self.state.elapsed_seconds,
            score_history=self.state.get_score_history_list(),
        )
