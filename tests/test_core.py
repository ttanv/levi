"""
Tests for core module: Program and EvaluationResult.

These are the fundamental data structures that everything else depends on.
"""

import pytest
from datetime import datetime

from algoforge.core import Program, EvaluationResult


class TestProgram:
    """Tests for Program dataclass."""

    def test_program_creation_minimal(self):
        """Program can be created with just code."""
        prog = Program(code="def solve(x): return x")

        assert prog.code == "def solve(x): return x"
        assert prog.id is not None
        assert len(prog.id) == 36  # UUID format
        assert prog.metadata == {}
        assert isinstance(prog.created_at, datetime)

    def test_program_creation_full(self):
        """Program can be created with all fields."""
        prog = Program(
            code="def solve(x): return x * 2",
            id="custom-id",
            metadata={"generation": 5, "island": 2},
        )

        assert prog.code == "def solve(x): return x * 2"
        assert prog.id == "custom-id"
        assert prog.metadata == {"generation": 5, "island": 2}

    def test_program_is_immutable(self):
        """Program is frozen (immutable)."""
        prog = Program(code="def solve(x): return x")

        with pytest.raises(AttributeError):
            prog.code = "new code"

    def test_program_unique_ids(self):
        """Each program gets a unique ID."""
        prog1 = Program(code="def solve(x): return x")
        prog2 = Program(code="def solve(x): return x")

        assert prog1.id != prog2.id

    def test_program_not_hashable_due_to_mutable_fields(self):
        """Programs are not hashable because metadata dict is mutable."""
        prog = Program(code="def solve(x): return x")

        # Program has mutable dict field (metadata), so it's not hashable
        # even though it's a frozen dataclass
        with pytest.raises(TypeError):
            hash(prog)

    def test_program_can_be_indexed_by_id(self):
        """Programs can be indexed by their unique id in dicts."""
        prog = Program(code="def solve(x): return x")

        # Use id as key instead
        d = {prog.id: prog}

        assert d[prog.id] == prog


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_evaluation_result_minimal(self):
        """EvaluationResult can be created with defaults."""
        result = EvaluationResult()

        assert result.scores == {}
        assert result.is_valid is True
        assert result.error is None

    def test_evaluation_result_full(self):
        """EvaluationResult can be created with all fields."""
        result = EvaluationResult(
            scores={"accuracy": 0.95, "speed": 0.8},
            is_valid=True,
            error=None,
        )

        assert result.scores == {"accuracy": 0.95, "speed": 0.8}
        assert result.is_valid is True
        assert result.error is None

    def test_evaluation_result_invalid(self):
        """EvaluationResult correctly represents invalid results."""
        result = EvaluationResult(
            is_valid=False,
            error="NameError: undefined variable 'x'",
        )

        assert result.is_valid is False
        assert result.error == "NameError: undefined variable 'x'"
        assert result.scores == {}

    def test_primary_score_with_score_key(self):
        """primary_score returns 'score' key when present."""
        result = EvaluationResult(
            scores={"score": 0.75, "accuracy": 0.8, "speed": 0.6}
        )

        assert result.primary_score == 0.75

    def test_primary_score_without_score_key(self):
        """primary_score returns first value when no 'score' key."""
        result = EvaluationResult(
            scores={"accuracy": 0.8, "speed": 0.6}
        )

        # Returns first value (dict order is preserved in Python 3.7+)
        assert result.primary_score == 0.8

    def test_primary_score_empty(self):
        """primary_score returns 0.0 when no scores."""
        result = EvaluationResult()

        assert result.primary_score == 0.0

    def test_invalid_factory_method(self):
        """invalid() factory creates invalid result correctly."""
        result = EvaluationResult.invalid(
            error="Timeout after 10s"
        )

        assert result.is_valid is False
        assert result.error == "Timeout after 10s"
        assert result.scores == {}

    def test_evaluation_result_mutable(self):
        """EvaluationResult is mutable (not frozen)."""
        result = EvaluationResult()

        # Should not raise
        result.scores = {"score": 0.5}
        assert result.scores == {"score": 0.5}


class TestProgramEvaluationIntegration:
    """Integration tests for Program + EvaluationResult."""

    def test_multiple_programs_multiple_results(self):
        """Multiple programs can have separate results."""
        prog1 = Program(code="def solve(x): return x")
        prog2 = Program(code="def solve(x): return x * 2")

        result1 = EvaluationResult(scores={"score": 0.5})
        result2 = EvaluationResult(scores={"score": 0.8})

        assert result1.primary_score == 0.5
        assert result2.primary_score == 0.8
