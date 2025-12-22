"""
BudgetManager: Tracks and enforces resource constraints.

Budget enforcement is by construction - operations check budget before proceeding.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from threading import Lock
import time

from .exceptions import BudgetExhausted


class ResourceType(Enum):
    """Types of resources that can be budgeted."""
    LLM_TOKENS = "llm_tokens"
    LLM_COST = "llm_cost"
    EVALUATIONS = "evaluations"
    WALL_TIME = "wall_time"


@dataclass
class BudgetManager:
    """
    Tracks and enforces resource constraints across the optimization run.

    Budget enforcement is by construction - primitives that consume resources
    take a BudgetManager and call try_consume() which raises BudgetExhausted
    if limits would be exceeded.
    """

    # Limits (None = unlimited)
    max_llm_tokens: Optional[int] = None
    max_llm_cost: Optional[float] = None
    max_evaluations: Optional[int] = None
    max_wall_time: Optional[float] = None  # seconds

    # Internal state
    _consumed_llm_tokens: int = field(default=0, repr=False)
    _consumed_llm_cost: float = field(default=0.0, repr=False)
    _consumed_evaluations: int = field(default=0, repr=False)
    _start_time: float = field(default_factory=time.time, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def consume(self, resource_type: ResourceType, amount: float) -> bool:
        """
        Consume resources. Returns True if successful, False if would exceed budget.
        """
        with self._lock:
            if self._would_exceed(resource_type, amount):
                return False
            self._apply_consumption(resource_type, amount)
            return True

    def try_consume(self, resource_type: ResourceType, amount: float) -> None:
        """
        Consume resources or raise BudgetExhausted.

        This is the primary method for budget-by-construction enforcement.
        """
        if not self.consume(resource_type, amount):
            raise BudgetExhausted(resource_type, self.remaining(resource_type), amount)

    def remaining(self, resource_type: ResourceType) -> Optional[float]:
        """Returns remaining budget for resource type (None = unlimited)."""
        if resource_type == ResourceType.LLM_TOKENS:
            if self.max_llm_tokens is None:
                return None
            return self.max_llm_tokens - self._consumed_llm_tokens

        elif resource_type == ResourceType.LLM_COST:
            if self.max_llm_cost is None:
                return None
            return self.max_llm_cost - self._consumed_llm_cost

        elif resource_type == ResourceType.EVALUATIONS:
            if self.max_evaluations is None:
                return None
            return self.max_evaluations - self._consumed_evaluations

        elif resource_type == ResourceType.WALL_TIME:
            if self.max_wall_time is None:
                return None
            elapsed = time.time() - self._start_time
            return self.max_wall_time - elapsed

        return None

    def is_exhausted(self) -> bool:
        """Returns True if any budget limit has been reached."""
        for rt in ResourceType:
            if self._is_limit_reached(rt):
                return True
        return False

    def check_budget(self) -> None:
        """Raises BudgetExhausted if any budget is depleted."""
        for rt in ResourceType:
            if self._is_limit_reached(rt):
                raise BudgetExhausted(rt, self.remaining(rt))

    def _would_exceed(self, resource_type: ResourceType, amount: float) -> bool:
        """Check if consuming amount would exceed the limit."""
        remaining = self.remaining(resource_type)
        if remaining is None:
            return False
        return amount > remaining

    def _is_limit_reached(self, resource_type: ResourceType) -> bool:
        """Check if a resource limit has been reached."""
        remaining = self.remaining(resource_type)
        if remaining is None:
            return False
        return remaining <= 0

    def _apply_consumption(self, resource_type: ResourceType, amount: float) -> None:
        """Apply the consumption (assumes lock is held)."""
        if resource_type == ResourceType.LLM_TOKENS:
            self._consumed_llm_tokens += int(amount)
        elif resource_type == ResourceType.LLM_COST:
            self._consumed_llm_cost += amount
        elif resource_type == ResourceType.EVALUATIONS:
            self._consumed_evaluations += int(amount)
        # WALL_TIME is not consumed, it's tracked by elapsed time

    @property
    def elapsed_time(self) -> float:
        """Returns elapsed wall time in seconds."""
        return time.time() - self._start_time
