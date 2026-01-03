"""Pipeline orchestration: coordinates producers and consumers."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import AlgoforgeConfig, AlgoforgeResult
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
    ):
        self.config = config
        self.pool = pool
        self.executor = executor
        self.archive_lock = asyncio.Lock()
        self.code_queue = asyncio.Queue()
        self.state = PipelineState(config.budget)
        self.stop_event = asyncio.Event()

        # Snapshot output directory
        self.output_dir: Optional[Path] = None
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

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

            # Final snapshot on completion
            if self.output_dir:
                self.save_snapshot(final=True)

        return self._build_result()

    async def _wait_for_completion(self) -> None:
        while not self.state.budget_exhausted:
            await asyncio.sleep(1.0)

    async def _status_monitor(self) -> None:
        while not self.stop_event.is_set():
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
