"""
ProgramPool Protocol: The method-differentiating primitive.

Different optimization methods (FunSearch, AlphaEvolve, GEPA) are distinguished
primarily by their ProgramPool implementation.
"""

from dataclasses import dataclass, field
from typing import Protocol, Optional, runtime_checkable

from ..core import Program, EvaluationResult


@dataclass
class SampleResult:
    """Result of sampling from a ProgramPool."""

    parent: Program
    inspirations: list[Program] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ProgramPool(Protocol):
    """
    Protocol for population management.

    The ProgramPool is the method-differentiating primitive. Different methods
    are distinguished primarily by their ProgramPool implementation.
    """

    def add(self, program: Program, evaluation_result: EvaluationResult) -> bool:
        """
        Add a program to the pool.

        Returns True if accepted, False if rejected by acceptance policy.
        """
        ...

    def sample(self, context: Optional[dict] = None) -> SampleResult:
        """
        Sample parent and inspiration programs for mutation.

        Args:
            context: Method-specific context for sampling decisions

        Returns:
            SampleResult with parent, inspirations, and sampling metadata
        """
        ...

    def best(self, metric: str = "score") -> Program:
        """Return the best program by the given metric."""
        ...

    def size(self) -> int:
        """Return the number of programs in the pool."""
        ...

    def on_generation_complete(self) -> None:
        """Called after each generation/iteration."""
        ...

    def on_epoch(self) -> None:
        """Called periodically (e.g., for island culling)."""
        ...
