"""Shared pipeline state for producer-consumer coordination."""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import BudgetConfig

logger = logging.getLogger(__name__)


class BudgetLimitReached(RuntimeError):
    """Raised when a new operation cannot start due to exhausted budget."""


# ---------------------------------------------------------------------------
# Module-level utilities (previously static methods on PipelineState)
# ---------------------------------------------------------------------------

def coerce_finite_float(value: object, *, default: float) -> float:
    """Best-effort numeric coercion with a safe fallback."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _coerce_positive_limit(value: object) -> float | None:
    """Normalize a budget limit.  Returns *None* for unlimited."""
    if value is None:
        return None
    return coerce_finite_float(value, default=0.0)


# ---------------------------------------------------------------------------
# Score history
# ---------------------------------------------------------------------------

@dataclass
class ScoreHistoryEntry:
    """A single entry in the score history."""
    eval_number: int
    score: float
    best_score: float
    timestamp: float
    accepted: bool
    sampler: str
    archive_size: int
    cell_index: int | None = None  # Which cell this evaluation fell into
    is_punctuated_equilibrium: bool = False  # Whether from PE
    cumulative_cost: float = 0.0


# ---------------------------------------------------------------------------
# BudgetTracker – budget limits, cost accounting, eval reservation
# ---------------------------------------------------------------------------

@dataclass
class BudgetTracker:
    """Tracks budget consumption and enforces limits.

    Owns every counter that participates in the budget-exhaustion decision
    (dollars, evaluations, seconds) plus the async lock that protects
    atomic reserve-if-budget-permits operations.
    """

    budget: BudgetConfig
    start_time: float = field(default_factory=time.time)

    # Cost tracking
    total_cost: float = 0.0

    # Eval / LLM counters
    eval_count: int = 0
    eval_in_flight: int = 0
    llm_in_flight: int = 0

    # Async lock – protects counter mutations that must be atomic with
    # budget-exhaustion checks (eval reservation, LLM gating).
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # LLM cost EMA – used by LLMGate to decide serial-mode
    _llm_cost_ema: float = 0.0
    _llm_cost_samples: int = 0

    # ---- budget queries ----

    @property
    def exhausted(self) -> bool:
        dollars_limit = _coerce_positive_limit(self.budget.dollars)
        if dollars_limit is not None:
            if dollars_limit <= 0.0:
                return True
            total_cost = coerce_finite_float(self.total_cost, default=float("inf"))
            if total_cost >= dollars_limit:
                return True

        if self.budget.evaluations is not None:
            eval_limit = int(coerce_finite_float(self.budget.evaluations, default=0.0))
            if eval_limit <= 0:
                return True
            eval_used = int(coerce_finite_float(
                self.eval_count + self.eval_in_flight, default=float("inf"),
            ))
            if eval_used >= eval_limit:
                return True

        seconds_limit = _coerce_positive_limit(self.budget.seconds)
        if seconds_limit is not None:
            if seconds_limit <= 0.0:
                return True
            if self.elapsed_seconds >= seconds_limit:
                return True

        return False

    @property
    def elapsed_seconds(self) -> float:
        start = coerce_finite_float(self.start_time, default=float("-inf"))
        elapsed = time.time() - start
        if not math.isfinite(elapsed):
            return float("inf")
        return max(0.0, elapsed)

    @property
    def progress(self) -> float:
        """Return progress as fraction (0-1) based on primary budget type."""
        dollars_limit = _coerce_positive_limit(self.budget.dollars)
        if dollars_limit is not None:
            if dollars_limit <= 0.0:
                return 1.0
            total_cost = coerce_finite_float(self.total_cost, default=float("inf"))
            return max(0.0, min(1.0, total_cost / dollars_limit))

        if self.budget.evaluations is not None:
            eval_limit = int(coerce_finite_float(self.budget.evaluations, default=0.0))
            if eval_limit <= 0:
                return 1.0
            eval_used = int(coerce_finite_float(
                self.eval_count + self.eval_in_flight, default=float("inf"),
            ))
            return max(0.0, min(1.0, eval_used / eval_limit))

        seconds_limit = _coerce_positive_limit(self.budget.seconds)
        if seconds_limit is not None:
            if seconds_limit <= 0.0:
                return 1.0
            return max(0.0, min(1.0, self.elapsed_seconds / seconds_limit))

        return 0.0

    # ---- cost recording ----

    def add_cost(self, cost: float) -> None:
        normalized = coerce_finite_float(cost, default=0.0)
        if normalized < 0.0:
            logger.warning("[Budget] Ignoring negative cost update: %r", cost)
            return
        self.total_cost += normalized

    def record_llm_cost(self, cost: object) -> None:
        """Record an LLM call cost (updates total_cost *and* EMA)."""
        normalized = coerce_finite_float(cost, default=0.0)
        if normalized < 0.0:
            return
        self.total_cost += normalized
        if self._llm_cost_samples == 0:
            self._llm_cost_ema = normalized
        else:
            self._llm_cost_ema = (0.8 * self._llm_cost_ema) + (0.2 * normalized)
        self._llm_cost_samples += 1

    # ---- eval reservation ----

    async def try_start_evaluation(self) -> bool:
        """Reserve one evaluation slot if budget permits."""
        async with self._lock:
            if self.exhausted:
                return False
            self.eval_in_flight += 1
            return True

    async def finish_evaluation(self) -> None:
        """Release one reserved evaluation slot."""
        async with self._lock:
            if self.eval_in_flight > 0:
                self.eval_in_flight -= 1

    # ---- serial-mode decision helpers (used by LLMGate) ----

    def remaining_dollars(self) -> float | None:
        dollars_limit = _coerce_positive_limit(self.budget.dollars)
        if dollars_limit is None:
            return None
        return dollars_limit - coerce_finite_float(self.total_cost, default=float("inf"))

    def _llm_serial_threshold(self, dollars_limit: float) -> float:
        if self._llm_cost_samples > 0:
            ema = self._llm_cost_ema
        else:
            ema = max(0.01 * dollars_limit, 0.05)
        return max(3.0 * ema, 0.03 * dollars_limit, 0.05)

    def should_use_serial_mode(self) -> bool:
        dollars_limit = _coerce_positive_limit(self.budget.dollars)
        if dollars_limit is not None:
            remaining = self.remaining_dollars()
            if remaining is not None and remaining <= self._llm_serial_threshold(dollars_limit):
                return True

        if self.budget.evaluations is not None:
            eval_limit = int(coerce_finite_float(self.budget.evaluations, default=0.0))
            eval_remaining = eval_limit - (self.eval_count + self.eval_in_flight)
            if eval_remaining <= 2:
                return True

        seconds_limit = _coerce_positive_limit(self.budget.seconds)
        if seconds_limit is not None:
            time_remaining = seconds_limit - self.elapsed_seconds
            if time_remaining <= 15.0:
                return True

        return False


# ---------------------------------------------------------------------------
# LLMGate – concurrency control and budget enforcement for LLM calls
# ---------------------------------------------------------------------------

class LLMGate:
    """Concurrency and budget gate for LLM API calls.

    Wraps every ``llm_client.acompletion`` call with:
    * semaphore-based concurrency limiting
    * budget-exhaustion check (raises ``BudgetLimitReached``)
    * automatic serial mode when budget is tight
    * cost extraction and accounting
    """

    def __init__(self, tracker: BudgetTracker) -> None:
        self._tracker = tracker
        self._semaphore = asyncio.Semaphore(1)
        self._serial_lock = asyncio.Lock()

    def configure_concurrency(self, max_in_flight: int) -> None:
        """Set global max concurrent LLM requests for this run."""
        limit = int(coerce_finite_float(max_in_flight, default=1.0))
        if limit <= 0:
            limit = 1
        self._semaphore = asyncio.Semaphore(limit)

    async def acompletion(
        self,
        llm_client: Any,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **extras: Any,
    ) -> Any:
        """Budget/concurrency gate around ``llm_client.acompletion``."""
        tracker = self._tracker
        async with self._semaphore:
            async with tracker._lock:
                if tracker.exhausted:
                    raise BudgetLimitReached("Budget exhausted before LLM call")
                tracker.llm_in_flight += 1
                use_serial = tracker.should_use_serial_mode()

            try:
                if use_serial:
                    async with self._serial_lock:
                        response = await llm_client.acompletion(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout=timeout,
                            **extras,
                        )
                else:
                    response = await llm_client.acompletion(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        **extras,
                    )
            finally:
                async with tracker._lock:
                    tracker.llm_in_flight = max(0, tracker.llm_in_flight - 1)

            async with tracker._lock:
                tracker.record_llm_cost(getattr(response, "cost", 0.0))

            return response


# ---------------------------------------------------------------------------
# PipelineState – thin coordinator that composes the above
# ---------------------------------------------------------------------------

class PipelineState:
    """Shared pipeline state for producer-consumer coordination.

    Composes :class:`BudgetTracker` (budget/cost enforcement) and
    :class:`LLMGate` (LLM concurrency control) with domain-specific
    pipeline state (eval metrics, meta-advice, score history).

    Delegation properties preserve the original flat API so that
    existing consumers (producer, consumer, runner, diversifier,
    equilibrium) require no changes.
    """

    def __init__(self, budget: BudgetConfig, start_time: float | None = None):
        st = start_time if start_time is not None else time.time()
        self.budget_tracker = BudgetTracker(budget, start_time=st)
        self.llm_gate = LLMGate(self.budget_tracker)

        # Eval outcome counters (not budget-relevant, so kept here)
        self.accept_count: int = 0
        self.error_count: int = 0

        # Meta-advice tracking
        self.current_meta_advice: str = ''
        self.previous_meta_advice: str = ''
        self.meta_advice_eval_count: int = 0

        # Period metrics for meta-advice generation
        self.period_errors: int = 0
        self.period_acceptances: int = 0
        self.period_rejections: int = 0
        self.period_error_messages: set = set()
        self.all_error_counts: dict = {}

        # Score history tracking
        self.score_history: list = []
        self.best_score_so_far: float = float('-inf')

        # Punctuated Equilibrium tracking
        self.last_pe_eval_count: int = 0
        self.pe_trigger_count: int = 0

    # ------------------------------------------------------------------
    # Delegation: BudgetTracker
    # ------------------------------------------------------------------

    @property
    def budget(self) -> BudgetConfig:
        return self.budget_tracker.budget

    @property
    def start_time(self) -> float:
        return self.budget_tracker.start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        self.budget_tracker.start_time = value

    @property
    def total_cost(self) -> float:
        return self.budget_tracker.total_cost

    @total_cost.setter
    def total_cost(self, value: float) -> None:
        self.budget_tracker.total_cost = value

    @property
    def eval_count(self) -> int:
        return self.budget_tracker.eval_count

    @eval_count.setter
    def eval_count(self, value: int) -> None:
        self.budget_tracker.eval_count = value

    @property
    def eval_in_flight(self) -> int:
        return self.budget_tracker.eval_in_flight

    @property
    def llm_in_flight(self) -> int:
        return self.budget_tracker.llm_in_flight

    @property
    def budget_exhausted(self) -> bool:
        return self.budget_tracker.exhausted

    @property
    def elapsed_seconds(self) -> float:
        return self.budget_tracker.elapsed_seconds

    @property
    def budget_progress(self) -> float:
        return self.budget_tracker.progress

    def add_cost(self, cost: float) -> None:
        self.budget_tracker.add_cost(cost)

    async def try_start_evaluation(self) -> bool:
        return await self.budget_tracker.try_start_evaluation()

    async def finish_evaluation(self) -> None:
        await self.budget_tracker.finish_evaluation()

    # ------------------------------------------------------------------
    # Delegation: LLMGate
    # ------------------------------------------------------------------

    def configure_llm_concurrency(self, max_in_flight: int) -> None:
        self.llm_gate.configure_concurrency(max_in_flight)

    async def acompletion(
        self,
        llm_client: Any,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **extras: Any,
    ) -> Any:
        return await self.llm_gate.acompletion(
            llm_client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **extras,
        )

    # ------------------------------------------------------------------
    # Backward-compat static helpers (used by runner / equilibrium)
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_finite_float(value: object, *, default: float) -> float:
        return coerce_finite_float(value, default=default)

    @staticmethod
    def _coerce_positive_limit(value: object) -> float | None:
        return _coerce_positive_limit(value)

    def _should_use_llm_serial_mode(self) -> bool:
        return self.budget_tracker.should_use_serial_mode()

    def _remaining_dollars(self) -> float | None:
        return self.budget_tracker.remaining_dollars()

    # ------------------------------------------------------------------
    # Domain: eval outcome recording
    # ------------------------------------------------------------------

    def record_accept(self) -> None:
        self.budget_tracker.eval_count += 1
        self.accept_count += 1
        self.period_acceptances += 1

    def record_reject(self) -> None:
        self.budget_tracker.eval_count += 1
        self.period_rejections += 1

    def record_error(self, error: str) -> None:
        self.budget_tracker.eval_count += 1
        self.error_count += 1
        self.period_errors += 1
        short_error = error[:100]
        if len(self.period_error_messages) < 10:
            self.period_error_messages.add(short_error)
        self.all_error_counts[short_error] = self.all_error_counts.get(short_error, 0) + 1

    # ------------------------------------------------------------------
    # Domain: meta-advice
    # ------------------------------------------------------------------

    def should_generate_meta_advice(self, interval: int) -> bool:
        if interval <= 0:
            return False
        return (
            self.eval_count > 0
            and self.eval_count % interval == 0
            and self.eval_count != self.meta_advice_eval_count
        )

    def reset_period_metrics(self) -> dict:
        top_errors = sorted(self.all_error_counts.items(), key=lambda x: -x[1])[:10]
        metrics = {
            'errors': self.period_errors,
            'acceptances': self.period_acceptances,
            'rejections': self.period_rejections,
            'error_messages': set(self.period_error_messages),
            'top_errors': top_errors,
        }
        self.period_errors = 0
        self.period_acceptances = 0
        self.period_rejections = 0
        self.period_error_messages.clear()
        self.meta_advice_eval_count = self.eval_count
        return metrics

    # ------------------------------------------------------------------
    # Domain: score history
    # ------------------------------------------------------------------

    def record_score(
        self,
        score: float,
        accepted: bool,
        sampler: str,
        archive_size: int,
        cell_index: int | None = None,
        is_punctuated_equilibrium: bool = False,
    ) -> None:
        """Record a score in the history."""
        if score > self.best_score_so_far:
            self.best_score_so_far = score

        entry = ScoreHistoryEntry(
            eval_number=self.eval_count,
            score=score,
            best_score=self.best_score_so_far,
            timestamp=time.time() - self.start_time,
            accepted=accepted,
            sampler=sampler,
            archive_size=archive_size,
            cell_index=cell_index,
            is_punctuated_equilibrium=is_punctuated_equilibrium,
            cumulative_cost=self.total_cost,
        )
        self.score_history.append(entry)

    def get_score_history_list(self) -> list[float]:
        """Get just the best scores over time for the result."""
        return [entry.best_score for entry in self.score_history]
