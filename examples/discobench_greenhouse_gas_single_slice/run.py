#!/usr/bin/env python3
"""Run LEVI on DiscoBench GreenhouseGasPrediction single-slice."""

from __future__ import annotations

from datetime import datetime
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault(
    "TIKTOKEN_CACHE_DIR",
    str(Path(__file__).resolve().parent / ".cache" / "tiktoken"),
)

import levi
from problem import (
    FUNCTION_SIGNATURE,
    PROBLEM_DESCRIPTION,
    SEED_PROGRAM,
    SLICE,
    TARGET_FUNCTION_NAME,
    TEST_DATASETS,
    TRAIN_DATASETS,
    score_fn,
)

DEFAULT_MODEL = "openrouter/qwen/qwen3-30b-a3b-instruct-2507"
PARADIGM_MODEL = os.environ.get("LEVI_GHG_PARADIGM_MODEL", os.environ.get("LEVI_GHG_MODEL", DEFAULT_MODEL))
MUTATION_MODEL = os.environ.get("LEVI_GHG_MUTATION_MODEL", os.environ.get("LEVI_GHG_MODEL", DEFAULT_MODEL))
LOCAL_ENDPOINT = os.environ.get("LEVI_GHG_LOCAL_ENDPOINT")


def _ensure_ghg_dependencies() -> None:
    missing: list[str] = []
    for module_name in ["discogen", "numpy", "sklearn", "typing_extensions"]:
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "GreenhouseGasPrediction example dependencies are not installed.\n"
            f"Missing modules: {joined}\n"
            "From the repo root, run:\n"
            "  uv sync\n"
            "  uv pip install discogen statsmodels torch typing-extensions\n"
            "Then rerun this example with:\n"
            "  uv run --no-sync python examples/discobench_greenhouse_gas_single_slice/run.py"
        )


def _load_target_callable(program_source: str):
    namespace: dict[str, object] = {
        "__name__": "__candidate__",
        "__source_code__": program_source,
    }
    exec(program_source, namespace)
    return namespace[TARGET_FUNCTION_NAME]


def _behavior_config() -> levi.BehaviorConfig:
    if SLICE == "model":
        return levi.BehaviorConfig(
            ast_features=[
                "function_def_count",
                "call_count",
                "comparison_count",
                "numeric_literal_count",
            ],
            score_keys=[
                "score_CH4",
                "score_SF6",
            ],
        )

    return levi.BehaviorConfig(
        ast_features=[
            "math_operators",
            "subscript_count",
            "call_count",
            "numeric_literal_count",
        ],
        score_keys=[
            "score_CH4",
            "score_SF6",
        ],
    )


def main() -> None:
    _ensure_ghg_dependencies()
    local_endpoints = {}
    if LOCAL_ENDPOINT:
        local_endpoints = {
            PARADIGM_MODEL: LOCAL_ENDPOINT,
            MUTATION_MODEL: LOCAL_ENDPOINT,
        }

    result = levi.evolve_code(
        PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        seed_program=SEED_PROGRAM,
        score_fn=score_fn,
        inputs=list(TRAIN_DATASETS),
        paradigm_model=PARADIGM_MODEL,
        mutation_model=MUTATION_MODEL,
        local_endpoints=local_endpoints,
        budget_evals=450,
        cvt=levi.CVTConfig(
            n_centroids=32,
        ),
        init=levi.InitConfig(
            n_diverse_seeds=5,
            n_variants_per_seed=5,
        ),
        behavior=_behavior_config(),
        pipeline=levi.PipelineConfig(
            n_llm_workers=4,
            n_eval_processes=4,
            eval_timeout=900,
        ),
        punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(
            enabled=True,
            interval=20,
            n_clusters=3,
            n_variants=2,
            behavior_noise=0.1,
        ),
        cascade=levi.CascadeConfig(
            enabled=False,
        ),
        output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_ghg_{SLICE}",
    )

    best_fn = _load_target_callable(result.best_program)
    held_out_metrics = score_fn(best_fn, list(TEST_DATASETS))

    print(f"\nSlice: {SLICE}")
    print(f"Best train score: {result.best_score:.6f}")
    if "score" in held_out_metrics:
        print(f"Held-out score: {held_out_metrics['score']:.6f}")
        print(f"Held-out mean Test MSE: {held_out_metrics['mean_test_mse']:.6f}")
    else:
        print(f"Held-out evaluation failed: {held_out_metrics.get('error', 'unknown error')}")
    print(result.best_program)


if __name__ == "__main__":
    main()
