"""
ProgramPool Protocol: The method-differentiating primitive.

Different optimization methods (FunSearch, AlphaEvolve, GEPA) are distinguished
primarily by their ProgramPool implementation.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from ..core import EvaluationResult, Program


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

    def add(self, program: Program, evaluation_result: EvaluationResult) -> tuple[bool, int]:
        """
        Add a program to the pool.

        Returns:
            Tuple of (accepted, cell_index). accepted is True if the program
            was added, False if rejected. cell_index is the slot/bin the program
            maps to (-1 if not applicable).
        """
        ...

    def sample(
        self,
        sampler_name: str,
        n_parents: int = 4,
        context: Optional[dict] = None,
    ) -> SampleResult:
        """
        Sample parent and inspiration programs for mutation.

        Args:
            sampler_name: Name of the sampling strategy to use
            n_parents: Number of parent/inspiration programs to sample
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
