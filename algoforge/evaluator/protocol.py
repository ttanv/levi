"""
Evaluator Protocol: Interface for program evaluation.

Responsible for sandboxed execution, timeout enforcement, and score computation.
"""

from dataclasses import dataclass, field
from typing import Protocol, Optional, Callable, runtime_checkable

from ..core import Program, EvaluationResult


@dataclass
class EvaluationStage:
    """A single stage in cascade evaluation."""

    name: str
    inputs: list
    timeout: float = 10.0
    validator: Optional[Callable[[EvaluationResult], bool]] = None


@runtime_checkable
class Evaluator(Protocol):
    """
    Protocol for program evaluation.

    Responsible for:
    - Sandboxed execution of untrusted generated code
    - Timeout and resource limit enforcement
    - Score computation from program outputs
    - Optional trace generation for reflective feedback
    """

    def evaluate(
        self,
        program: Program,
        inputs: list,
        generate_traces: bool = False
    ) -> EvaluationResult:
        """
        Evaluate a single program on given inputs.

        Args:
            program: The program to evaluate
            inputs: List of input values to test
            generate_traces: Whether to capture execution traces

        Returns:
            EvaluationResult with scores, outputs, validity, and optional traces
        """
        ...

    def evaluate_cascade(
        self,
        program: Program,
        stages: list[EvaluationStage]
    ) -> Optional[EvaluationResult]:
        """
        Evaluate program through progressive stages.

        Early terminates if any stage fails validation.

        Returns:
            Final EvaluationResult if all stages pass, None if rejected early
        """
        ...
