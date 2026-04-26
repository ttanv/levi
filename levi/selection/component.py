"""Per-component selection strategies for bundle prompt evolution.

Selectors decide *which* prompt component in a bundle should be mutated at
each evolutionary step. They receive per-target accept/reject feedback and
adapt over time.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union


class ComponentSelector(ABC):
    """Abstract per-component selector."""

    @abstractmethod
    def select(self, targets: Sequence[str], *, context: Optional[dict] = None) -> str:
        """Choose a target from the candidate list."""

    @abstractmethod
    def update(
        self,
        target: str,
        *,
        accepted: bool,
        score_gain: Optional[float] = None,
    ) -> None:
        """Record the outcome of a mutation targeting ``target``."""

    @abstractmethod
    def stats(self) -> dict[str, dict[str, float]]:
        """Return per-target counters for logging/debugging."""


class _StatsStore:
    """Shared accept/total counters used by UCB and stagnation selectors."""

    def __init__(self) -> None:
        self._n_samples: dict[str, int] = {}
        self._n_successes: dict[str, int] = {}
        self._total: int = 0

    def _ensure(self, target: str) -> None:
        if target not in self._n_samples:
            self._n_samples[target] = 0
            self._n_successes[target] = 0

    def record(self, target: str, accepted: bool) -> None:
        self._ensure(target)
        self._n_samples[target] += 1
        if accepted:
            self._n_successes[target] += 1
        self._total += 1

    def samples(self, target: str) -> int:
        return self._n_samples.get(target, 0)

    def successes(self, target: str) -> int:
        return self._n_successes.get(target, 0)

    def success_rate(self, target: str) -> float:
        n = self._n_samples.get(target, 0)
        if n == 0:
            return 0.5
        return self._n_successes[target] / n

    @property
    def total(self) -> int:
        return self._total

    def as_stats_dict(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for target, samples in self._n_samples.items():
            out[target] = {
                "n_samples": float(samples),
                "n_successes": float(self._n_successes.get(target, 0)),
                "success_rate": self.success_rate(target),
            }
        return out


class RoundRobinComponentSelector(ComponentSelector):
    """Deterministic cycle across the provided target list."""

    def __init__(self) -> None:
        self._cursor: int = 0
        self._stats = _StatsStore()

    def select(self, targets: Sequence[str], *, context: Optional[dict] = None) -> str:
        if not targets:
            raise ValueError("RoundRobinComponentSelector.select requires at least one target")
        target = targets[self._cursor % len(targets)]
        self._cursor += 1
        return target

    def update(
        self,
        target: str,
        *,
        accepted: bool,
        score_gain: Optional[float] = None,
    ) -> None:
        del score_gain
        self._stats.record(target, accepted)

    def stats(self) -> dict[str, dict[str, float]]:
        return self._stats.as_stats_dict()


class UCBComponentSelector(ComponentSelector):
    """UCB1 over per-component acceptance rate.

    Unvisited components score +inf and therefore get a warmup round
    before any exploitation kicks in. Reward = 1 on accepted offspring,
    0 otherwise.
    """

    def __init__(self, c: float = 2.0) -> None:
        self.c = c
        self._stats = _StatsStore()

    def _ucb(self, target: str) -> float:
        n = self._stats.samples(target)
        if n == 0:
            return float("inf")
        exploitation = self._stats.success_rate(target)
        exploration = self.c * math.sqrt(math.log(self._stats.total + 1) / n)
        return exploitation + exploration

    def select(self, targets: Sequence[str], *, context: Optional[dict] = None) -> str:
        if not targets:
            raise ValueError("UCBComponentSelector.select requires at least one target")
        best_target = targets[0]
        best_score = self._ucb(best_target)
        for target in targets[1:]:
            score = self._ucb(target)
            if score > best_score:
                best_score = score
                best_target = target
        return best_target

    def update(
        self,
        target: str,
        *,
        accepted: bool,
        score_gain: Optional[float] = None,
    ) -> None:
        del score_gain
        self._stats.record(target, accepted)

    def stats(self) -> dict[str, dict[str, float]]:
        return self._stats.as_stats_dict()


class StagnationComponentSelector(ComponentSelector):
    """Picks the least-improving component (inverse UCB).

    Intended for punctuated equilibrium: shift whichever component has
    the lowest acceptance rate in the main loop. Unvisited components
    have priority so every target gets at least one shift eventually.
    Optionally reads stats from a shared selector via context["main_stats"].
    """

    def __init__(self, c: float = 2.0) -> None:
        self.c = c
        self._stats = _StatsStore()

    def _score(
        self,
        target: str,
        shared_stats: Optional[dict[str, dict[str, float]]],
    ) -> float:
        # Prefer shared main-loop stats if provided.
        if shared_stats and target in shared_stats:
            n = int(shared_stats[target].get("n_samples", 0))
            rate = float(shared_stats[target].get("success_rate", 0.5))
            total = sum(int(row.get("n_samples", 0)) for row in shared_stats.values())
        else:
            n = self._stats.samples(target)
            rate = self._stats.success_rate(target)
            total = self._stats.total
        if n == 0:
            return float("inf")  # unseen -> pick first
        # Invert: low success_rate -> high score. Add exploration bonus so
        # rarely-visited components aren't starved even if their rate is high.
        exploitation = 1.0 - rate
        exploration = self.c * math.sqrt(math.log(total + 1) / n)
        return exploitation + exploration

    def select(self, targets: Sequence[str], *, context: Optional[dict] = None) -> str:
        if not targets:
            raise ValueError("StagnationComponentSelector.select requires at least one target")
        shared_stats = None
        if context is not None:
            shared_stats = context.get("main_stats")
        best_target = targets[0]
        best_score = self._score(best_target, shared_stats)
        for target in targets[1:]:
            score = self._score(target, shared_stats)
            if score > best_score:
                best_score = score
                best_target = target
        return best_target

    def update(
        self,
        target: str,
        *,
        accepted: bool,
        score_gain: Optional[float] = None,
    ) -> None:
        del score_gain
        self._stats.record(target, accepted)

    def stats(self) -> dict[str, dict[str, float]]:
        return self._stats.as_stats_dict()


SelectorSpec = Union[str, ComponentSelector]


def make_component_selector(spec: SelectorSpec) -> ComponentSelector:
    """Resolve a string name or pass through an existing selector."""
    if isinstance(spec, ComponentSelector):
        return spec
    if isinstance(spec, str):
        key = spec.lower()
        if key == "ucb":
            return UCBComponentSelector()
        if key == "round_robin":
            return RoundRobinComponentSelector()
        if key == "stagnation":
            return StagnationComponentSelector()
        raise ValueError(
            f"Unknown component_selector: {spec!r}. Use 'ucb', 'round_robin', 'stagnation', "
            "or pass a ComponentSelector instance."
        )
    raise TypeError(
        f"component_selector must be a string or ComponentSelector; got {type(spec).__name__}"
    )
