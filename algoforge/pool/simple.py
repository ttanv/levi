"""
SimplePool: Basic reference implementation of ProgramPool.

Uses fitness-proportionate selection without behavioral diversity.
"""

import math
import random
from typing import Optional

from ..core import Program, EvaluationResult
from ..database import InMemoryStorage, StorageRecord
from .protocol import SampleResult


class SimplePool:
    """
    Simple reference implementation of ProgramPool.

    Uses softmax-weighted sampling based on scores.
    """

    def __init__(
        self,
        storage: Optional[InMemoryStorage] = None,
        max_size: Optional[int] = None,
        temperature: float = 1.0,
    ) -> None:
        """
        Args:
            storage: Storage backend (defaults to InMemoryStorage)
            max_size: Maximum pool size (prunes worst when exceeded)
            temperature: Sampling temperature (higher = more exploration)
        """
        self._storage = storage or InMemoryStorage()
        self._max_size = max_size
        self._temperature = temperature
        self._record_ids: list[str] = []

    def add(self, program: Program, evaluation_result: EvaluationResult) -> bool:
        """Add program if valid. Prunes worst if over max_size."""
        if not evaluation_result.is_valid:
            return False

        record_id = self._storage.insert(program, evaluation_result, {})
        self._record_ids.append(record_id)

        if self._max_size and len(self._record_ids) > self._max_size:
            self._prune_worst()

        return True

    def sample(self, context: Optional[dict] = None, n_parents: int = 2) -> SampleResult:
        """
        Sample parent programs using fitness-proportionate selection.

        Args:
            context: Optional sampling context
            n_parents: Number of parents to sample (default 2)
        """
        if not self._record_ids:
            raise ValueError("Pool is empty - cannot sample")

        records = self._storage.query()
        if not records:
            raise ValueError("Pool is empty - cannot sample")

        scores = [r.evaluation_result.primary_score for r in records]
        probs = self._softmax(scores)

        sampled_indices = []
        for _ in range(min(n_parents, len(records))):
            idx = random.choices(range(len(records)), weights=probs)[0]
            sampled_indices.append(idx)

        parent = records[sampled_indices[0]].program
        inspirations = [records[i].program for i in sampled_indices[1:]]

        return SampleResult(
            parent=parent,
            inspirations=inspirations,
            metadata={"sampled_ids": [records[i].id for i in sampled_indices]},
        )

    def best(self, metric: str = "score") -> Program:
        """Return highest-scoring program."""
        records = self._storage.query(order_by=f"-{metric}", limit=1)
        if not records:
            raise ValueError("Pool is empty")
        return records[0].program

    def size(self) -> int:
        """Return number of programs in pool."""
        return len(self._record_ids)

    def on_generation_complete(self) -> None:
        """Called after each generation (no-op for SimplePool)."""
        pass

    def on_epoch(self) -> None:
        """Called periodically (no-op for SimplePool)."""
        pass

    def _softmax(self, scores: list[float]) -> list[float]:
        """Compute softmax probabilities with temperature."""
        if not scores:
            return []
        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / self._temperature) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    def _prune_worst(self) -> None:
        """Remove lowest-scoring program."""
        records = self._storage.query(order_by="score", limit=1)
        if records:
            self._storage.remove(records[0].id)
            self._record_ids.remove(records[0].id)
