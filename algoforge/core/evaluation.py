"""
EvaluationResult: The outcome of evaluating a program.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from .types import MetricDict, OutputDict


@dataclass
class EvaluationResult:
    """
    Attributes:
        program_id: Reference to the evaluated program
        scores: Dictionary mapping metric names to scalar values
        outputs: Dictionary of program outputs per test input (called "signature" in funsearch paper)
        is_valid: Boolean indicating whether the program executed correctly
        eval_time_seconds: Time taken for evaluation
        traces: Optional execution traces, logs, or diagnostic text
        error: Error message if not valid
    """

    program_id: str
    scores: MetricDict = field(default_factory=dict)
    outputs: OutputDict = field(default_factory=dict)
    is_valid: bool = True
    eval_time_seconds: float = 0.0
    traces: Optional[str] = None
    error: Optional[str] = None

    @property
    def primary_score(self) -> float:
        """Returns the primary score ('score' key, or first available, or 0.0)."""
        if 'score' in self.scores:
            return self.scores['score']
        if self.scores:
            return next(iter(self.scores.values()))
        return 0.0

    @classmethod
    def invalid(cls, program_id: str, error: str) -> 'EvaluationResult':
        """Factory method for creating an invalid result."""
        return cls(
            program_id=program_id,
            is_valid=False,
            error=error,
        )
