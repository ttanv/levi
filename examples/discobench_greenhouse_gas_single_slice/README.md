# DiscoBench Greenhouse Gas — single-slice LEVI search

Uses LEVI to search the official DiscoBench `GreenhouseGasPrediction`
single-slice benchmark. The example defaults to the `model.py` slice and uses
the official train/test split:

- meta-train: `CH4`, `SF6`
- held-out meta-test: `CO2`, `N2O`

It also supports the second official single-slice task:

```bash
LEVI_GHG_SLICE=data_processing uv run --no-sync python run.py
```

These are two different official DiscoBench tasks:

- `GreenhouseGasPrediction_model`: LEVI edits only `model.py`
- `GreenhouseGasPrediction_data_processing`: LEVI edits only `data_processing.py`

## Setup

```bash
uv sync
uv pip install discogen statsmodels torch typing-extensions
```

## Run

```bash
cd examples/discobench_greenhouse_gas_single_slice
uv run --no-sync python run.py
```

If you want to use a different model or a local OpenAI-compatible endpoint:

```bash
LEVI_GHG_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
LEVI_GHG_LOCAL_ENDPOINT=http://localhost:8000/v1 \
uv run --no-sync python run.py
```

## How it works

1. On first use, `problem.py` copies the official Greenhouse Gas templates and
   datasets out of the installed `discogen` package into `.task_cache/`.
2. LEVI mutates either `model.py` or `data_processing.py`, depending on
   `LEVI_GHG_SLICE`.
3. `score_fn` evaluates candidate code on the official training datasets
   (`CH4`, `SF6`) by running the real per-dataset `main.py` harness.
4. Scores are aggregated as mean relative improvement over the official
   baseline MSEs using a bounded score, plus temporal backtests on prefixes of
   the training series so the search is rewarded for extrapolation rather than
   only fitting the official training gases.
5. After evolution finishes, `run.py` also evaluates the best program on the
   held-out DiscoBench datasets (`CO2`, `N2O`).

## Archive Behavior

The LEVI archive uses both structural code features and per-dataset score
signals.

- For the `model` slice: `function_def_count`, `call_count`,
  `comparison_count`, `numeric_literal_count`, plus `score_CH4` and `score_SF6`
- For the `data_processing` slice: `math_operators`, `subscript_count`,
  `call_count`, `numeric_literal_count`, plus `score_CH4` and `score_SF6`

This keeps programs behaviorally distinct both by code shape and by how they
trade off performance across the two training gases.

## Default Search Config

The current runner uses:

- `budget_evals=320`
- `n_diverse_seeds=5`
- `n_variants_per_seed=5`
- `n_centroids=32`
- punctuated equilibrium enabled every 20 evaluations

## Notes

- `discogen` currently ships a `GreenhouseGasPrediction` `main.py` template
  with an undefined `train_mse` variable. This example patches that template in
  the local task cache before running evaluations.
- Use `LEVI_GHG_TASK_CACHE_DIR=/some/path` to move the generated cache.
- Use `GHG_DATASET_TIMEOUT_SECONDS=...` to change the per-dataset subprocess timeout.
