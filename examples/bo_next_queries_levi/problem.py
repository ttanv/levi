"""
DiscoBench BayesianOptimisation next_queries — LEVI problem definition.

Evaluates candidate next_queries implementations by running them inside
DiscoGen's actual BO loop via subprocess. The candidate source code is
written to each dataset's next_queries.py, then `python main.py` is
executed per dataset. Scores are aggregated across datasets.

Setup:
    uv sync
    uv pip install discogen
    cd examples/bo_next_queries_levi
    discogen create-task --task-domain BayesianOptimisation
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

TASK_DIR = Path(os.environ.get(
    "DISCOGEN_TASK_DIR",
    Path(__file__).parent / "task_src" / "BayesianOptimisation",
))
DATASET_TIMEOUT_SECONDS = int(os.environ.get("BO_DATASET_TIMEOUT_SECONDS", "900"))

ALL_DATASETS = [
    "Ackley1d", "Ackley2d", "Branin2d", "Bukin2d", "Cosine8d",
    "DropWave2d", "EggHolder2d", "Griewank5d", "Hartmann6d",
    "HolderTable2d", "Levy6d",
]

QUICK_CONFIG_OVERRIDES = {
    "surrogate_fit_posterior_num_steps": 200,
    "acq_sample_size": 2000,
    "acq_top_n_samples": 20,
    "acq_gradient_max_iter": 5,
    "fixed_budget": 8,
    "fixed_num_initial_samples": 4,
    "num_seeds": 1,
}
# These overrides define a cheaper proxy eval used for cascade screening.
# The proxy score is only compared against other proxy scores in the same
# predicted archive cell; final archive ranking still uses full scores.

# ── Problem description shown to the LLM ────────────────────────────

PROBLEM_DESCRIPTION = """
# Bayesian Optimisation — Next Query Selection

You are implementing the `next_queries` function for a Bayesian Optimisation
(BO) loop. This function is called at each iteration to select which candidate
point(s) to evaluate next on the objective function.

## Context

At each BO iteration, a surrogate model (Gaussian Process) has been fitted to
the observed data. An acquisition function (e.g. Expected Improvement) has been
evaluated on a large set of candidate points. Your function receives these
acquisition function values along with all observations so far, and must return
the next point(s) to query.

## What the baseline does

The default implementation simply sorts candidates by acquisition function value
(descending) and returns the top `batch_size` candidates. This is pure greedy
exploitation of the acquisition function.

## How to improve

The baseline ignores several inputs that could be useful:
- `obs_samples` / `obs_values`: the history of observed points and their values
- `remaining_budget`: how many evaluations are left
- `candidate_samples`: the spatial locations of candidates (not just their scores)

Better strategies might:
- Add diversity: penalise candidates that are too close to each other or to
  already-observed points, encouraging exploration of unvisited regions.
- Be budget-aware: explore more when budget is high, exploit more when low.
- Use softmax/stochastic selection instead of hard argmax for robustness.
- Normalise acquisition values (z-score, rank) before selection for scale
  invariance across iterations.
- Combine acquisition values with distance-based bonuses.

## Important constraints
- The function MUST use `jax.numpy` (imported as `jnp`), not regular numpy.
- The function MUST return a jnp.ndarray of shape (batch_size, D).
- `batch_size` comes from `config['next_queries_batch_size']` (usually 1).
- The function must be deterministic given the same inputs (use jax.random with
  a key derived from remaining_budget if you need randomness).
- The algorithm will be tested on held-out objective functions, so it must
  generalise — do not overfit to specific benchmark functions.
"""

FUNCTION_SIGNATURE = """
from typing import Any
import jax.numpy as jnp

def next_queries(obs_samples: jnp.ndarray,
                 obs_values: jnp.ndarray,
                 candidate_samples: jnp.ndarray,
                 candidate_acq_fn_vals: jnp.ndarray,
                 remaining_budget: int,
                 config: dict[str, Any]) -> jnp.ndarray:
    '''
    Select the next query point(s) for Bayesian Optimisation.

    Args:
        obs_samples: (N, D) previously observed input points.
        obs_values: (N,) corresponding objective function values.
        candidate_samples: (M, D) candidate points to choose from.
        candidate_acq_fn_vals: (M,) acquisition function values for each candidate.
        remaining_budget: number of objective function evaluations remaining.
        config: configuration dict, must read config['next_queries_batch_size'].

    Returns:
        (batch_size, D) array of selected query points.
    '''
    pass
"""

SEED_PROGRAM = """
from typing import Any
import jax.numpy as jnp

def next_queries(obs_samples: jnp.ndarray,
                 obs_values: jnp.ndarray,
                 candidate_samples: jnp.ndarray,
                 candidate_acq_fn_vals: jnp.ndarray,
                 remaining_budget: int,
                 config: dict[str, Any]) -> jnp.ndarray:
    '''Select the next query point(s) by greedy acquisition function maximisation.'''
    sorted_idxs = jnp.argsort(candidate_acq_fn_vals, descending=True)
    sorted_candidate_samples = candidate_samples[sorted_idxs]
    return sorted_candidate_samples[:config['next_queries_batch_size']]
"""

# ── Evaluation via DiscoGen subprocess ──────────────────────────────


def _run_dataset(dataset_dir: Path, timeout: int = DATASET_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Run a single dataset's main.py and parse the JSON output."""
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=str(dataset_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr_tail = result.stderr[-1000:] if result.stderr else ""
        stdout_tail = result.stdout[-1000:] if result.stdout else ""
        details = stderr_tail or stdout_tail or "non-zero exit"
        raise RuntimeError(details)

    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("no JSON in stdout")


def _patch_config_py(config_path: Path, overrides: dict[str, int]) -> None:
    """Apply simple scalar overrides to the copied dataset config."""
    config_text = config_path.read_text()
    for key, value in overrides.items():
        config_text, n = re.subn(rf"('{re.escape(key)}':\s*)\d+", rf"\g<1>{value}", config_text, count=1)
        if n != 1:
            raise RuntimeError(f"Could not patch config key: {key}")
    config_path.write_text(config_text)


def _parse_eval_spec(item: Any) -> tuple[str, str]:
    """Return (dataset_name, eval_mode) where mode is 'full' or 'quick'."""
    if isinstance(item, str):
        if item.startswith("quick:"):
            return item.split(":", 1)[1], "quick"
        if item.startswith("full:"):
            return item.split(":", 1)[1], "full"
        return item, "full"
    raise TypeError(f"Unsupported dataset spec: {item!r}")


def _run_dataset_isolated(
    dataset_dir: Path,
    code_str: str,
    timeout: int = DATASET_TIMEOUT_SECONDS,
    eval_mode: str = "full",
) -> dict[str, Any]:
    """Run a dataset from a private temp copy so parallel evals do not share files."""
    with tempfile.TemporaryDirectory(prefix=f"levi_bo_{dataset_dir.name}_") as tmp_dir:
        tmp_dataset_dir = Path(tmp_dir) / dataset_dir.name
        shutil.copytree(dataset_dir, tmp_dataset_dir)
        (tmp_dataset_dir / "next_queries.py").write_text(code_str)
        if eval_mode == "quick":
            _patch_config_py(tmp_dataset_dir / "config.py", QUICK_CONFIG_OVERRIDES)
        return _run_dataset(tmp_dataset_dir, timeout=timeout)


def score_fn(next_queries_fn: Any, inputs: list[str]) -> dict:
    """Evaluate a candidate next_queries against DiscoGen BO benchmarks.

    Runs each dataset inside its own temp copy, writes the candidate source
    code there, and aggregates scores. This keeps evals isolated so multiple
    LEVI eval workers can run safely in parallel.
    """
    code_str = next_queries_fn.__globals__.get("__source_code__")
    if not code_str:
        return {"error": "no source code available (missing __source_code__ in globals)"}

    datasets = inputs if inputs else ALL_DATASETS
    task_dir = TASK_DIR
    if not task_dir.is_dir():
        return {"error": f"task dir not found: {task_dir}. Run: discogen create-task --task-domain BayesianOptimisation"}

    scores: dict[str, float] = {}
    errors: dict[str, str] = {}
    start = time.perf_counter()

    for item in datasets:
        ds, eval_mode = _parse_eval_spec(item)
        ds_dir = task_dir / ds
        if not ds_dir.is_dir():
            scores[ds] = float("-inf")
            continue

        try:
            metrics = _run_dataset_isolated(ds_dir, code_str, eval_mode=eval_mode)
            scores[ds] = metrics["maximum_value_mean"]
        except Exception as e:
            scores[ds] = float("-inf")
            errors[ds] = str(e)

    elapsed = time.perf_counter() - start

    valid_scores = [v for v in scores.values() if v != float("-inf")]
    if not valid_scores:
        error_summary = "; ".join(f"{k}: {v}" for k, v in errors.items()) if errors else "unknown error"
        return {
            "error": f"all datasets failed: {error_summary}",
            **{f"max_{k}": v for k, v in scores.items()},
        }

    overall = sum(valid_scores) / len(valid_scores)
    return {
        "score": overall,
        "execution_time": elapsed,
        "datasets_ok": len(valid_scores),
        "datasets_total": len(datasets),
        **{f"max_{k}": v for k, v in scores.items()},
    }
