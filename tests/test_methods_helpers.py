"""Tests for low-level method helpers and utility primitives."""

import numpy as np
import pytest

from algoforge.core import EvaluationResult, Program
from algoforge.methods.algoforge import _restore_from_snapshot
from algoforge.utils import extract_fn_name, evaluate_code


class DummyExtractor:
    def __init__(self):
        self.phase = None

    def set_phase(self, phase: str):
        self.phase = phase


class DummyPool:
    def __init__(self, n_dims: int = 3, centroids=None):
        self._n_dims = n_dims
        self._centroids = centroids
        self._mins = None
        self._maxs = None
        self._ranges = None
        self.init_called = 0
        self.add_calls = []

    def _init_cvt_centroids(self):
        self.init_called += 1
        return np.zeros((4, self._n_dims), dtype=float)

    def add(self, program, eval_result):
        self.add_calls.append((program, eval_result))
        return True, 0


class TestEvaluateCode:
    def test_executes_valid_code_and_calls_score_fn(self):
        observed = {}

        def score_fn(fn, inputs):
            observed["name"] = fn.__name__
            return {"score": sum(fn(x) for x in inputs)}

        code = "def solve(x):\n    return x + 1"
        result = evaluate_code(code, score_fn, [1, 2, 3], "solve")

        assert result == {"score": 9}
        assert observed["name"] == "solve"

    def test_returns_syntax_error_dict(self):
        code = "def solve(x)\n    return x"
        result = evaluate_code(code, lambda *_: {"score": 0}, [1], "solve")

        assert "error" in result
        assert result["error"].startswith("Syntax error:")

    def test_returns_missing_function_error(self):
        code = "def other(x):\n    return x"
        result = evaluate_code(code, lambda *_: {"score": 0}, [1], "solve")

        assert result == {"error": "Function 'solve' not found (got NoneType)"}

    def test_catches_score_function_exceptions(self):
        def score_fn(_fn, _inputs):
            raise RuntimeError("scoring failed")

        code = "def solve(x):\n    return x"
        result = evaluate_code(code, score_fn, [1], "solve")

        assert "error" in result
        assert "scoring failed" in result["error"]


class TestRestoreFromSnapshot:
    def test_initializes_centroids_when_deferred_and_restores_elites(self):
        pool = DummyPool(n_dims=2, centroids=None)
        extractor = DummyExtractor()
        snapshot = {
            "run_state": {"total_cost": 3.5},
            "elites": [
                {"code": "def solve(x):\n    return x", "scores": {"score": 1.0}},
                {"code": "def solve(x):\n    return x + 1", "scores": {"score": 2.0}},
            ],
        }

        resumed_cost = _restore_from_snapshot(pool, extractor, snapshot)

        assert resumed_cost == 3.5
        assert pool.init_called == 1
        assert len(pool.add_calls) == 2
        assert isinstance(pool.add_calls[0][0], Program)
        assert isinstance(pool.add_calls[0][1], EvaluationResult)
        assert pool._mins.tolist() == [0.0, 0.0]
        assert pool._maxs.tolist() == [1.0, 1.0]
        assert pool._ranges.tolist() == [1.0, 1.0]
        assert extractor.phase == "evolution"

    def test_skips_centroid_init_when_already_present(self):
        pool = DummyPool(n_dims=2, centroids=np.ones((3, 2)))
        extractor = DummyExtractor()
        snapshot = {"run_state": {"total_cost": 0.2}, "elites": []}

        resumed_cost = _restore_from_snapshot(pool, extractor, snapshot)

        assert resumed_cost == 0.2
        assert pool.init_called == 0
        assert extractor.phase == "evolution"


class TestExtractFnName:
    def test_extracts_name_from_valid_signature(self):
        assert extract_fn_name("def schedule_txns(txns, budget):") == "schedule_txns"

    def test_falls_back_to_solve(self):
        assert extract_fn_name("schedule_txns(txns, budget)") == "solve"
