"""Tests for low-level method helpers and utility primitives."""

import asyncio

import numpy as np
import pytest

from levi.artifacts import CodeAdapter
from levi.behavior import BehaviorExtractor
from levi.config.models import BudgetConfig, InitConfig, LeviConfig
from levi.core import EvaluationResult, Program
from levi.init.diversifier import Diversifier
from levi.methods.levi import (
    _activate_proxy_benchmark_discovery_inputs,
    _evaluate_seed,
    _restore_from_snapshot,
    _restore_proxy_benchmark_state,
)
from levi.pipeline.state import PipelineState
from levi.utils import evaluate_code, evaluate_prompt, extract_fn_name


class DummyExtractor:
    pass


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


class TestEvaluatePrompt:
    def test_supports_prompt_evaluator_returning_number(self):
        def evaluator(prompt):
            return len(prompt)

        result = evaluate_prompt("Be more explicit.", evaluator, None)

        assert result == {"score": 17.0}

    def test_preserves_per_problem_prompt_metrics(self):
        def evaluator(prompt):
            return {"score": len(prompt) / 10.0, "correctness": [1.0, 0.0, 1.0]}

        result = evaluate_prompt("Be precise.", evaluator, None)

        assert result == {"score": 1.1, "correctness": [1.0, 0.0, 1.0]}

    def test_behavior_extractor_supports_score_only_for_non_code_content(self):
        extractor = BehaviorExtractor(ast_features=[], score_keys=["score"])
        extractor.set_fixed_bounds({"score": (0.0, 10.0)})

        behavior = extractor.extract(Program(content="Please solve carefully."), {"score": 2.5})

        assert behavior["score"] == 0.25


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

    def test_restores_using_content_field_when_present(self):
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
                    "content": "def solve(x):\n    return x + 3",
                    "scores": {"score": 6.0},
                    "behavior": {"loop_count": 0.25, "branch_count": 0.75},
                }
            ],
        }
        config = _make_diversifier_config()

        resumed_cost = _restore_from_snapshot(pool, extractor, snapshot, artifact_adapter=CodeAdapter(config))

        assert resumed_cost == 1.25
        assert pool.add_at_cell_calls[0][1].content == "def solve(x):\n    return x + 3"


class TestExtractFnName:
    def test_extracts_name_from_valid_signature(self):
        assert extract_fn_name("def schedule_txns(txns, budget):") == "schedule_txns"

    def test_falls_back_to_solve(self):
        assert extract_fn_name("schedule_txns(txns, budget)") == "solve"


class TestProxyBenchmarkState:
    def test_activate_proxy_benchmark_discovery_inputs_overrides_runtime_inputs(self):
        config = LeviConfig(
            problem_description="Test problem",
            function_signature="def solve(x):",
            score_fn=_dummy_score_fn,
            budget=BudgetConfig(evaluations=10),
            inputs=["small"],
            proxy_benchmark={"enabled": True, "discovery_inputs": ["full_a", "full_b"]},
        )

        _activate_proxy_benchmark_discovery_inputs(config)

        assert config.inputs == ["full_a", "full_b"]

    def test_restore_proxy_benchmark_state_applies_selected_indices(self):
        config = LeviConfig(
            problem_description="Test problem",
            function_signature="def solve(x):",
            score_fn=_dummy_score_fn,
            budget=BudgetConfig(evaluations=10),
            proxy_benchmark={"enabled": True, "discovery_inputs": ["p0", "p1", "p2"]},
        )
        snapshot = {"metadata": {"proxy_benchmark": {"selected_indices": [2, 0]}}}

        _restore_proxy_benchmark_state(config, snapshot)

        assert config.proxy_benchmark.selected_indices == [2, 0]
        assert config.inputs == ["p2", "p0"]


def _dummy_score_fn(_fn, _inputs=None):
    return {"score": 1.0}


def _make_diversifier_config(*, budget: BudgetConfig | None = None, init: InitConfig | None = None) -> LeviConfig:
    return LeviConfig(
        problem_description="Test problem",
        function_signature="def solve(x):",
        score_fn=_dummy_score_fn,
        budget=budget or BudgetConfig(evaluations=10),
        init=init or InitConfig(n_diverse_seeds=0, n_variants_per_seed=0),
    )


class _AsyncExecutor:
    def __init__(self, result: dict | None = None):
        self.result = result or {"score": 1.0}
        self.calls = 0

    async def run(self, *_args, **_kwargs):
        self.calls += 1
        return dict(self.result)


class _AsyncArtifactAdapter:
    artifact_type = "prompt"

    def __init__(self, *, result: dict | None = None, exc: Exception | None = None):
        self.result = result or {"score": 1.0}
        self.exc = exc
        self.calls = 0

    async def evaluate(self, _executor, _content, **_kwargs):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return dict(self.result)


class TestSeedEvaluation:
    def test_evaluate_seed_returns_none_on_error_result(self):
        config = LeviConfig(
            problem_description="Test problem",
            function_signature="def solve(x):",
            score_fn=_dummy_score_fn,
            seed_program="seed prompt",
            budget=BudgetConfig(evaluations=10),
        )
        state = PipelineState(config.budget)
        adapter = _AsyncArtifactAdapter(result={"error": "transient api failure"})

        result = asyncio.run(_evaluate_seed(config, object(), state, adapter))

        assert result is None
        assert adapter.calls == 1
        assert state.error_count == 1

    def test_evaluate_seed_returns_none_on_exception(self):
        config = LeviConfig(
            problem_description="Test problem",
            function_signature="def solve(x):",
            score_fn=_dummy_score_fn,
            seed_program="seed prompt",
            budget=BudgetConfig(evaluations=10),
        )
        state = PipelineState(config.budget)
        adapter = _AsyncArtifactAdapter(exc=RuntimeError("boom"))

        result = asyncio.run(_evaluate_seed(config, object(), state, adapter))

        assert result is None
        assert adapter.calls == 1
        assert state.error_count == 1


class TestDiversifier:
    def test_generate_diverse_seeds_retries_three_times_when_generation_fails(self, monkeypatch):
        config = _make_diversifier_config(init=InitConfig(n_diverse_seeds=0, n_variants_per_seed=0))
        state = PipelineState(config.budget)
        diversifier = Diversifier(config, _AsyncExecutor(), CodeAdapter(config), state)
        attempts = []

        class _Resp:
            content = "not code"
            cost = 0.0

        async def fake_completion(**kwargs):
            attempts.append(kwargs["model"])
            return _Resp()

        monkeypatch.setattr(diversifier, "_acompletion", fake_completion)

        seeds = asyncio.run(diversifier._generate_diverse_seeds(None, None))

        assert seeds == []
        assert len(attempts) == 3

    def test_cascade_eval_uses_reserved_eval_slot(self):
        config = _make_diversifier_config(budget=BudgetConfig(evaluations=1))
        executor = _AsyncExecutor()
        state = PipelineState(config.budget)
        diversifier = Diversifier(config, executor, CodeAdapter(config), state)

        assert asyncio.run(state.try_start_evaluation()) is True
        result = asyncio.run(diversifier._cascade_eval("def solve(x):\n    return x"))
        asyncio.run(state.finish_evaluation())

        assert result == {"score": 1.0}
        assert executor.calls == 1
