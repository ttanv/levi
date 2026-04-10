"""
EvaluationResult: The outcome of evaluating a program.
"""

from dataclasses import dataclass, field
from typing import Optional

from .types import MetricDict


@dataclass
class EvaluationResult:
    """
    Attributes:
        scores: Dictionary mapping metric names to scalar values
        is_valid: Boolean indicating whether the program executed correctly
        error: Error message if not valid
    """

    scores: MetricDict = field(default_factory=dict)
    is_valid: bool = True
    error: Optional[str] = None

    @property
    def primary_score(self) -> float:
        """Returns the primary score ('score' key, or first available, or 0.0)."""
        if "score" in self.scores:
            return self.scores["score"]
        if self.scores:
            return next(iter(self.scores.values()))
        return 0.0
