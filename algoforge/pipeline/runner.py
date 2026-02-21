"""Pipeline orchestration: coordinates producers and consumers."""

import asyncio
import json
import logging
import math
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
        init_score_history: Optional[list] = None,
        start_time: Optional[float] = None,
        state: Optional[PipelineState] = None,
    ):
        self.config = config
        self.pool = pool
        self.executor = executor
        self.archive_lock = asyncio.Lock()
        self.code_queue = asyncio.Queue()
        self.state = state or PipelineState(config.budget)
        if start_time is not None:
            self.state.start_time = start_time
        if state is None:
            # Include init phase cost when runner owns state lifecycle.
            self.state.total_cost = self.state._coerce_finite_float(init_cost, default=0.0)
        initial_best = pool.get_stats().get("best_score", float("-inf"))
        self.state.best_score_so_far = self.state._coerce_finite_float(initial_best, default=float("-inf"))
        if init_score_history and not self.state.score_history:
            self.state.score_history = list(init_score_history)
            self.state.eval_count = len(init_score_history)
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
                state=self.state,
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
        while True:
            if self.state.budget_exhausted:
                break
            await asyncio.sleep(1.0)

    def _sync_best_score_from_pool(self) -> None:
        """Keep pipeline state best score aligned with archive best score."""
        pool_best = self.pool.get_stats().get("best_score", float("-inf"))
        try:
            normalized = float(pool_best)
        except (TypeError, ValueError):
            return
        if not math.isfinite(normalized):
            return
        if normalized > self.state.best_score_so_far:
            self.state.best_score_so_far = normalized

    def _ingest_pe_stats(self, stats: dict) -> None:
        """Apply PE outcomes to state counters/history and line-by-line logs."""
        evaluations = stats.get("evaluations", [])
        for eval_data in evaluations:
            source = eval_data.get("source", "pe")
            model = str(eval_data.get("model", "unknown")).split("/")[-1]
            archive_size = eval_data.get("archive_size", self.pool.size())
            try:
                archive_size_int = int(archive_size)
            except (TypeError, ValueError):
                archive_size_int = self.pool.size()
            cell_index = eval_data.get("cell_index")

            if "error" in eval_data:
                error_msg = str(eval_data.get("error", "unknown error"))
                self.state.record_error(error_msg)
                logger.info(
                    f"[Eval #{self.state.eval_count}] PE/{model:27s} ERROR ({source}): {error_msg[:80]}"
                )
                continue

            try:
                score = float(eval_data.get("score", 0.0))
            except (TypeError, ValueError):
                error_msg = f"Invalid PE score for {model}: {eval_data.get('score')!r}"
                self.state.record_error(error_msg)
                logger.info(f"[Eval #{self.state.eval_count}] PE/{model:27s} ERROR: {error_msg[:80]}")
                continue
            if not math.isfinite(score):
                error_msg = f"Non-finite PE score for {model}: {eval_data.get('score')!r}"
                self.state.record_error(error_msg)
                logger.info(f"[Eval #{self.state.eval_count}] PE/{model:27s} ERROR: {error_msg[:80]}")
                continue

            accepted = bool(eval_data.get("accepted", False))
            if accepted:
                self.state.record_accept()
            else:
                self.state.record_reject()

            is_new_best = score > self.state.best_score_so_far
            try:
                cell_index_int = int(cell_index) if cell_index is not None else None
            except (TypeError, ValueError):
                cell_index_int = None

            self.state.record_score(
                score=score,
                accepted=accepted,
                sampler=f"pe_{source}",
                archive_size=archive_size_int,
                cell_index=cell_index_int,
                is_punctuated_equilibrium=True,
            )

            if is_new_best:
                status = "NEW BEST ★"
            elif accepted:
                status = "accepted"
            else:
                status = "rejected"

            logger.info(
                f"[Eval #{self.state.eval_count}] PE/{model:27s} {status:12s} | "
                f"score: {score:.1f} | best: {self.state.best_score_so_far:.1f} | "
                f"${self.state.total_cost:.3f}"
            )

    async def _status_monitor(self) -> None:
        while not self.stop_event.is_set():
            try:
                await asyncio.sleep(30.0)
                best_score = self.pool.get_stats().get("best_score", float("-inf"))
                if not isinstance(best_score, (int, float)) or not math.isfinite(float(best_score)):
                    best_score = float("-inf")
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
        if pe_config.interval <= 0:
            logger.warning("[PE] Disabled: interval must be > 0")
            return

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
                        if not isinstance(stats, dict):
                            raise ValueError(f"Unexpected PE stats type: {type(stats).__name__}")
                        # PE LLM cost is accounted centrally by state.acompletion.
                        self._ingest_pe_stats(stats)
                        self._sync_best_score_from_pool()
                        # Prevent immediate re-triggering when PE's own evaluations
                        # move eval_count onto another interval boundary.
                        self.state.last_pe_eval_count = max(
                            self.state.last_pe_eval_count,
                            self.state.eval_count,
                        )
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
                "cumulative_cost": entry.cumulative_cost,
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
            best_program=best.content if best else "",
            best_score=self.pool.get_stats().get("best_score", 0.0),
            total_evaluations=self.state.eval_count,
            total_cost=self.state.total_cost,
            archive_size=self.pool.size(),
            runtime_seconds=self.state.elapsed_seconds,
            score_history=self.state.get_score_history_list(),
        )
