from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_problem_module():
    pytest.importorskip("discogen")

    module_path = (
        Path(__file__).resolve().parent.parent
        / "examples"
        / "discobench_greenhouse_gas_single_slice"
        / "problem.py"
    )
    spec = importlib.util.spec_from_file_location("ghg_problem_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_slice_seed_program_matches_baseline(monkeypatch):
    monkeypatch.setenv("LEVI_GHG_SLICE", "model")
    problem = _load_problem_module()

    namespace = {
        "__name__": "__candidate__",
        "__source_code__": problem.SEED_PROGRAM,
    }
    exec(problem.SEED_PROGRAM, namespace)
    fn = namespace[problem.TARGET_FUNCTION_NAME]

    metrics = problem.score_fn(fn, list(problem.TRAIN_DATASETS))

    assert metrics["datasets_ok"] == 2
    assert metrics["slice"] == "model"
    assert metrics["score"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["test_mse_CH4"] == pytest.approx(problem.BASELINE_TEST_MSE["CH4"], rel=1e-9, abs=1e-6)
    assert metrics["test_mse_SF6"] == pytest.approx(problem.BASELINE_TEST_MSE["SF6"], rel=1e-9, abs=1e-6)
