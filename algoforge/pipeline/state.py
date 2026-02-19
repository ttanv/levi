"""Shared pipeline state for producer-consumer coordination."""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from ..config import BudgetConfig

logger = logging.getLogger(__name__)


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


@dataclass
class PipelineState:
    budget: BudgetConfig
    start_time: float = field(default_factory=time.time)

    # Cost tracking
    total_cost: float = 0.0

    # Eval counters
    eval_count: int = 0
    accept_count: int = 0
    error_count: int = 0

    # In-flight tracking
    llm_in_flight: int = 0
    eval_in_flight: int = 0

    # Meta-advice tracking
    current_meta_advice: str = ''
    previous_meta_advice: str = ''
    meta_advice_eval_count: int = 0

    # Period metrics for meta-advice generation
    period_errors: int = 0
    period_acceptances: int = 0
    period_rejections: int = 0
    period_error_messages: set = field(default_factory=set)

    all_error_counts: dict = field(default_factory=dict)

    # Score history tracking
    score_history: list = field(default_factory=list)
    best_score_so_far: float = float('-inf')

    # Punctuated Equilibrium tracking
    last_pe_eval_count: int = 0
    pe_trigger_count: int = 0

    @staticmethod
    def _coerce_finite_float(value: object, *, default: float) -> float:
        """Best-effort numeric coercion with a safe fallback."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(numeric):
            return default
        return numeric

    @staticmethod
    def _coerce_positive_limit(value: object) -> float | None:
        """Normalize a budget limit. Returns None for unlimited."""
        if value is None:
            return None
        limit = PipelineState._coerce_finite_float(value, default=0.0)
        return limit

    @property
    def budget_exhausted(self) -> bool:
        dollars_limit = self._coerce_positive_limit(self.budget.dollars)
        if dollars_limit is not None:
            if dollars_limit <= 0.0:
                return True
            total_cost = self._coerce_finite_float(self.total_cost, default=float("inf"))
            if total_cost >= dollars_limit:
                return True

        if self.budget.evaluations is not None:
            eval_limit = int(self._coerce_finite_float(self.budget.evaluations, default=0.0))
            if eval_limit <= 0:
                return True
            eval_count = int(self._coerce_finite_float(self.eval_count, default=float("inf")))
            if eval_count >= eval_limit:
                return True

        seconds_limit = self._coerce_positive_limit(self.budget.seconds)
        if seconds_limit is not None:
            if seconds_limit <= 0.0:
                return True
            if self.elapsed_seconds >= seconds_limit:
                return True

        return False

    @property
    def elapsed_seconds(self) -> float:
        start_time = self._coerce_finite_float(self.start_time, default=float("-inf"))
        elapsed = time.time() - start_time
        if not math.isfinite(elapsed):
            return float("inf")
        return max(0.0, elapsed)

    @property
    def budget_progress(self) -> float:
        """Return progress as fraction (0 to 1) based on primary budget type."""
        dollars_limit = self._coerce_positive_limit(self.budget.dollars)
        if dollars_limit is not None:
            if dollars_limit <= 0.0:
                return 1.0
            total_cost = self._coerce_finite_float(self.total_cost, default=float("inf"))
            return max(0.0, min(1.0, total_cost / dollars_limit))

        if self.budget.evaluations is not None:
            eval_limit = int(self._coerce_finite_float(self.budget.evaluations, default=0.0))
            if eval_limit <= 0:
                return 1.0
            eval_count = int(self._coerce_finite_float(self.eval_count, default=float("inf")))
            return max(0.0, min(1.0, eval_count / eval_limit))

        seconds_limit = self._coerce_positive_limit(self.budget.seconds)
        if seconds_limit is not None:
            if seconds_limit <= 0.0:
                return 1.0
            return max(0.0, min(1.0, self.elapsed_seconds / seconds_limit))

        return 0.0

    def add_cost(self, cost: float) -> None:
        normalized_cost = self._coerce_finite_float(cost, default=0.0)
        if normalized_cost < 0.0:
            logger.warning("[Budget] Ignoring negative cost update: %r", cost)
            return
        self.total_cost += normalized_cost

    def record_accept(self) -> None:
        self.eval_count += 1
        self.accept_count += 1
        self.period_acceptances += 1

    def record_reject(self) -> None:
        self.eval_count += 1
        self.period_rejections += 1

    def record_error(self, error: str) -> None:
        self.eval_count += 1
        self.error_count += 1
        self.period_errors += 1
        short_error = error[:100]
        if len(self.period_error_messages) < 10:
            self.period_error_messages.add(short_error)
        self.all_error_counts[short_error] = self.all_error_counts.get(short_error, 0) + 1

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
