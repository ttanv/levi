#!/usr/bin/env python3
"""Run LEVI on DiscoBench BayesianOptimisation next_queries."""

import os
from pathlib import Path

os.environ.setdefault(
    "TIKTOKEN_CACHE_DIR",
    str(Path(__file__).resolve().parent / ".cache" / "tiktoken"),
)

import levi
from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, score_fn

# ── Dataset subset ──────────────────────────────────────────────────
# Single dataset with a cheaper quick proxy. In the main evolution loop this
# is used for per-cell cascade screening; full archive ranking still uses the
# full eval score.
DATASETS = ["full:Ackley1d"]
QUICK_INPUTS = ["quick:Ackley1d"]
# Full (all 11):
# DATASETS = [
#     "Ackley1d", "Ackley2d", "Branin2d", "Bukin2d", "Cosine8d",
#     "DropWave2d", "EggHolder2d", "Griewank5d", "Hartmann6d",
#     "HolderTable2d", "Levy6d",
# ]

PARADIGM_MODEL = "openrouter/qwen/qwen3-30b-a3b-instruct-2507"
MUTATION_MODEL = "openrouter/qwen/qwen3-30b-a3b-instruct-2507"


def _ensure_bo_dependencies() -> None:
    try:
        import jax  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name == "jax":
            raise SystemExit(
                "BayesianOptimisation task dependencies are not installed.\n"
                "From the repo root, run:\n"
                "  uv sync\n"
                "  cd examples/bo_next_queries_levi/task_src/BayesianOptimisation\n"
                "  bash install.sh\n"
                "Then rerun this example with:\n"
                "  uv run --no-sync python run.py"
            ) from e
        raise
    except RuntimeError as e:
        if "jaxlib version" in str(e) and "incompatible with jax version" in str(e):
            raise SystemExit(
                "BayesianOptimisation has an incompatible JAX install in the repo's uv environment.\n"
                "Reinstall the task dependencies with matched versions:\n"
                "  cd examples/bo_next_queries_levi/task_src/BayesianOptimisation\n"
                "  bash install.sh\n"
                "Then rerun this example with:\n"
                "  cd ../../..\n"
                "  uv run --no-sync python examples/bo_next_queries_levi/run.py\n"
                "If you need the CUDA build instead of CPU, set:\n"
                "  LEVI_BO_JAX_TARGET=cuda12 bash install.sh"
            ) from e
        raise


def main() -> None:
    _ensure_bo_dependencies()
    result = levi.evolve_code(
        PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        seed_program=SEED_PROGRAM,
        score_fn=score_fn,
        inputs=DATASETS,
        paradigm_model=PARADIGM_MODEL,
        mutation_model=MUTATION_MODEL,
        budget_evals=200,
        init=levi.InitConfig(
            n_diverse_seeds=4,
            n_variants_per_seed=5,
        ),
        pipeline=levi.PipelineConfig(
            n_llm_workers=4,
            n_eval_processes=4,  # safe now: each eval runs in isolated temp dataset copies
            eval_timeout=1200,  # 3 datasets * 300s each, plus buffer for setup/overhead
        ),
        punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(
            enabled=False,
            interval=5,
        ),
        cascade=levi.CascadeConfig(
            enabled=False,
            quick_inputs=QUICK_INPUTS,
            min_score_ratio=1.0,  # Only run full eval when the target cell's quick proxy is matched or improved.
            quick_timeout=180,
        ),
    )

    print(f"\nBest score: {result.best_score:.6f}")
    print(result.best_program)


if __name__ == "__main__":
    main()
