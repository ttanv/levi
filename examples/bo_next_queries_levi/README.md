# DiscoBench BO — next_queries LEVI search

Uses LEVI to discover better `next_queries` implementations for DiscoGen's
BayesianOptimisation domain. Evaluation runs against DiscoGen's actual BO loop.

## Setup

```bash
uv sync
uv pip install discogen

# Create the task (from this directory)
cd examples/bo_next_queries_levi
discogen create-task --task-domain BayesianOptimisation

# Install DiscoGen's task-specific deps
cd task_src/BayesianOptimisation && bash install.sh && cd ../..
```

## Run

```bash
uv run --no-sync python run.py
```

Edit `run.py` to change the dataset subset, model, or budget.

## How it works

1. LEVI generates candidate `next_queries` implementations via LLM mutations.
2. `score_fn` writes each candidate's source code to the DiscoGen task
   directory and runs the BO loop via subprocess.
3. Scores (mean maximum objective value across seeds) are aggregated over
   the selected datasets.
4. In the main evolution loop, if cascade is enabled, the quick proxy is used
   to decide whether a candidate should receive a full eval in its predicted
   archive cell.
5. LEVI evolves toward higher-scoring strategies based on full eval scores.

## Notes

- `n_eval_processes` can be >1 since evaluations run in isolated temp copies.
- Set `DISCOGEN_TASK_DIR` env var to override the default task path.
- The function interface uses `jax.numpy` to match DiscoGen exactly.
- The example's `quick:...` inputs are a cheaper BO proxy. They are compared
  against the incumbent quick score in the same target cell during the main
  evolution loop, not against the run's best full score.
- `install.sh` uses `uv pip` and installs a matched JAX/JAXLIB pair into the
  repo's `uv` environment.
- The default JAX target is CPU, which is the safe path on macOS and for the
  Python 3.12 DiscoGen setup. Set `LEVI_BO_JAX_TARGET=cuda12` only if you
  explicitly want the CUDA 12 JAX build.
- Use `uv run --no-sync` for this example after installing task-specific deps,
  otherwise `uv run` will try to restore the root lockfile and can fight the BO
  environment.
