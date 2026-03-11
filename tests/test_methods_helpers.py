"""Tests for low-level method helpers and utility primitives."""

import numpy as np
import pytest

from levi.core import EvaluationResult, Program
from levi.methods.levi import _restore_from_snapshot
from levi.utils import evaluate_code, extract_fn_name


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
        self._n_centroids = 0
        self.init_called = 0
        self.add_calls = []
        self.add_at_cell_calls = []

    def _init_cvt_centroids(self):
        self.init_called += 1
        return np.zeros((4, self._n_dims), dtype=float)

    def add(self, program, eval_result):
        self.add_calls.append((program, eval_result))
        return True, 0

    def add_at_cell(self, cell_index, program, eval_result, behavior):
        self.add_at_cell_calls.append((cell_index, program, eval_result, behavior))
        return True


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

    def test_supports_score_fn_without_inputs(self):
        observed = {}

        def score_fn(fn):
            observed["name"] = fn.__name__
            return {"score": fn(4)}

        code = "def solve(x):\n    return x + 3"
        result = evaluate_code(code, score_fn, None, "solve")

        assert result == {"score": 7}
        assert observed["name"] == "solve"

    def test_supports_score_fn_without_inputs_even_when_inputs_provided(self):
        def score_fn(fn):
            return {"score": fn(10)}

        code = "def solve(x):\n    return x - 2"
        result = evaluate_code(code, score_fn, [1, 2, 3], "solve")

        assert result == {"score": 8}

    def test_two_arg_score_fn_receives_empty_inputs_when_config_inputs_missing(self):
        observed = {}

        def score_fn(_fn, inputs):
            observed["inputs"] = inputs
            return {"score": float(len(inputs))}

        code = "def solve(x):\n    return x"
        result = evaluate_code(code, score_fn, None, "solve")

        assert result == {"score": 0.0}
        assert observed["inputs"] == []


class TestRestoreFromSnapshot:
    def test_requires_snapshot_geometry(self):
        pool = DummyPool(n_dims=2, centroids=None)
        extractor = DummyExtractor()
        snapshot = {
            "run_state": {"total_cost": 3.5},
            "elites": [
                {"code": "def solve(x):\n    return x", "scores": {"score": 1.0}},
                {"code": "def solve(x):\n    return x + 1", "scores": {"score": 2.0}},
            ],
        }

        with pytest.raises(KeyError):
            _restore_from_snapshot(pool, extractor, snapshot)

    def test_restores_centroids_bounds_and_cell_behavior_when_present(self):
        pool = DummyPool(n_dims=2, centroids=None)
        extractor = DummyExtractor()
        snapshot = {
            "metadata": {
                "centroids": [[0.1, 0.2], [0.8, 0.9]],
                "normalization": {
                    "mins": [0.0, 0.0],
                    "maxs": [10.0, 20.0],
                    "ranges": [10.0, 20.0],
                },
            },
            "run_state": {"total_cost": 1.25},
            "elites": [
                {
                    "cell_index": 1,
                    "code": "def solve(x):\n    return x + 2",
                    "scores": {"score": 5.0},
                    "behavior": {"loop_count": 0.25, "branch_count": 0.75},
                    "metadata": {"tag": "restored"},
                }
            ],
        }

        resumed_cost = _restore_from_snapshot(pool, extractor, snapshot)

        assert resumed_cost == 1.25
        assert pool.init_called == 0
        assert pool._n_centroids == 2
        assert pool._centroids.tolist() == [[0.1, 0.2], [0.8, 0.9]]
        assert pool._mins.tolist() == [0.0, 0.0]
        assert pool._maxs.tolist() == [10.0, 20.0]
        assert pool._ranges.tolist() == [10.0, 20.0]
        assert len(pool.add_at_cell_calls) == 1
        assert len(pool.add_calls) == 0
        assert pool.add_at_cell_calls[0][0] == 1
        assert isinstance(pool.add_at_cell_calls[0][1], Program)
        assert pool.add_at_cell_calls[0][1].metadata == {"tag": "restored"}
        assert isinstance(pool.add_at_cell_calls[0][2], EvaluationResult)
        assert extractor.phase == "evolution"

    def test_requires_cell_and_behavior_per_elite(self):
        pool = DummyPool(n_dims=2, centroids=None)
        extractor = DummyExtractor()
        snapshot = {
            "metadata": {
                "centroids": [[0.1, 0.2], [0.8, 0.9]],
                "normalization": {
                    "mins": [0.0, 0.0],
                    "maxs": [10.0, 20.0],
                    "ranges": [10.0, 20.0],
                },
            },
            "run_state": {"total_cost": 1.25},
            "elites": [
                {
                    "code": "def solve(x):\n    return x + 2",
                    "scores": {"score": 5.0},
                    "metadata": {"tag": "restored"},
                }
            ],
        }

        with pytest.raises(KeyError):
            _restore_from_snapshot(pool, extractor, snapshot)


class TestExtractFnName:
    def test_extracts_name_from_valid_signature(self):
        assert extract_fn_name("def schedule_txns(txns, budget):") == "schedule_txns"

    def test_falls_back_to_solve(self):
        assert extract_fn_name("schedule_txns(txns, budget)") == "solve"
