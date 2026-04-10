"""
DiscoBench GreenhouseGasPrediction single-slice LEVI problem definition.

This example targets the official DiscoBench GreenhouseGasPrediction
single-slice benchmark splits:

- train datasets: CH4, SF6
- held-out datasets: CO2, N2O

The editable slice defaults to `model.py`, but can be switched to
`data_processing.py` with `LEVI_GHG_SLICE=data_processing`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SLICE = os.environ.get("LEVI_GHG_SLICE", "model").strip().lower()
if SLICE not in {"model", "data_processing"}:
    raise ValueError("LEVI_GHG_SLICE must be one of: model, data_processing")

TARGET_FUNCTION_NAME = "make_model" if SLICE == "model" else "process_data"
TARGET_MODULE_FILE = f"{SLICE}.py"

TRAIN_DATASETS = ("CH4", "SF6")
TEST_DATASETS = ("CO2", "N2O")
ALL_DATASETS = TRAIN_DATASETS + TEST_DATASETS

TASK_CACHE_DIR = Path(
    os.environ.get(
        "LEVI_GHG_TASK_CACHE_DIR",
        str(Path(__file__).resolve().parent / ".task_cache" / f"GreenhouseGasPrediction_{SLICE}"),
    )
)
DATASET_TIMEOUT_SECONDS = int(os.environ.get("GHG_DATASET_TIMEOUT_SECONDS", "300"))

def _build_problem_assets() -> tuple[str, str, str]:
    if SLICE == "model":
        description = """
# Greenhouse Gas Prediction — DiscoBench Single Slice (`model.py`)

You are implementing the official DiscoBench `GreenhouseGasPrediction_model`
single-slice problem.

## Benchmark split

- Meta-train datasets: `CH4`, `SF6`
- Held-out meta-test datasets: `CO2`, `N2O`
- Editable file: `model.py`
- Fixed file: `data_processing.py`

## Domain

Each dataset is a greenhouse gas concentration time series with a future
forecast horizon. The gases differ in scale, cadence, and dynamics:

- `CH4`: monthly methane, strong long-term trend and seasonality
- `SF6`: monthly sulfur hexafluoride, mostly smooth secular growth
- `CO2`: daily carbon dioxide, strong seasonal cycle plus long-term trend
- `N2O`: monthly nitrous oxide, smoother long-term growth

Your code must generalize across these differences rather than overfit to one
specific gas or cadence.

## Evaluation harness

For each dataset:

1. Load `train_data.npy` and `test_data.npy`, both shaped `(N, 5)`.
2. Apply the fixed `process_data(data)` function.
3. Call your `make_model(pre_processed_data)` function.
4. Fit the returned model with:
   `model.fit(processed[:, :-2], processed[:, -1])`
5. Predict with:
   `model.predict(processed[:, :-2])`
6. Measure held-out `Test MSE`.

The search score is the negative mean future-horizon `Test MSE` across the
training gases `CH4` and `SF6`. Lower test MSE is always better.

## Interface requirements

Your returned object must expose:

- `predicts_std: bool`
- `fit(X, y) -> self`
- `predict(X) -> np.ndarray`

You may use `numpy`, `scikit-learn`, `statsmodels`, `torch`, and the Python
standard library. Keep the interface exact.

## Guidance

- Favor models that extrapolate trends cleanly.
- Handle both monthly and daily periodic structure when possible.
- Avoid hard-coding dataset-specific constants or train/test boundaries.
- The first three feature columns are temporal covariates, but their exact
  semantics/order differ across gases. Do not assume column 0 always means the
  same thing across datasets.
- The held-out evaluation uses unseen gases, so transfer matters more than
  squeezing a tiny gain from one training dataset, but the optimizer itself is
  driven only by train-gas test MSE.
"""

        signature = """
from typing import Any

import numpy as np

def make_model(pre_processed_data: np.ndarray) -> Any:
    '''
    Build a forecasting model from the already-processed greenhouse gas data.

    Args:
        pre_processed_data: Array shaped (N, 5) after fixed preprocessing.

    Returns:
        A model object with:
        - `predicts_std: bool`
        - `fit(X, y) -> self`
        - `predict(X) -> np.ndarray`
    '''
    pass
"""

        seed_program = """from typing import Protocol

import numpy as np
from sklearn.linear_model import LinearRegression
from typing_extensions import Self

class Model(Protocol):
    predicts_std: bool

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        pass


def make_model(pre_processed_data: np.ndarray) -> Model:
    class LinRegModel(LinearRegression):
        predicts_std = False

    return LinRegModel()
"""
        return description, signature, seed_program

    description = """
# Greenhouse Gas Prediction — DiscoBench Single Slice (`data_processing.py`)

You are implementing the official DiscoBench
`GreenhouseGasPrediction_data_processing` single-slice problem.

## Benchmark split

- Meta-train datasets: `CH4`, `SF6`
- Held-out meta-test datasets: `CO2`, `N2O`
- Editable file: `data_processing.py`
- Fixed file: `model.py` (baseline linear regression)

## Evaluation harness

For each dataset:

1. Load raw `train_data.npy` and `test_data.npy`, both shaped `(N, 5)`.
2. Apply your `process_data(data)` function.
3. Fit the fixed baseline model with:
   `model.fit(processed[:, :-2], processed[:, -1])`
4. Predict with:
   `model.predict(processed[:, :-2])`
5. Measure held-out `Test MSE`.

Important: the harness uses all columns except the final two as features, and
uses the final column as the target. The penultimate column is ignored by the
fixed training script. Your preprocessing must respect that exact contract.

The search score is the negative mean future-horizon `Test MSE` across the
training gases `CH4` and `SF6`. Lower test MSE is always better.

## Interface requirements

- Return a numeric `np.ndarray`.
- Keep the target in the final column.
- Ensure the returned array works for both train and test datasets.

## Guidance

- Engineer features that help a fixed linear model extrapolate trend and seasonality.
- Monthly and daily datasets both appear, so avoid assumptions tied to only one cadence.
- You may need to canonicalize temporal columns so that downstream modeling is
  less sensitive to dataset-specific column ordering.
- Because the model slice is fixed, feature design is your main lever.
"""

    signature = """
import numpy as np

def process_data(data: np.ndarray) -> np.ndarray:
    '''
    Preprocess greenhouse gas time-series data before the fixed model is fitted.

    Args:
        data: Raw array shaped (N, 5).

    Returns:
        A processed numeric array where the final column is the prediction
        target and all feature columns come before the final two columns.
    '''
    pass
"""

    seed_program = """import numpy as np

def process_data(data: np.ndarray) -> np.ndarray:
    return data
"""
    return description, signature, seed_program


PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM = _build_problem_assets()


def _discogen_domain_root() -> Path:
    try:
        import discogen
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "discogen is not installed. Install it in the repo environment before using this example."
        ) from e

    import inspect

    return Path(inspect.getfile(discogen)).resolve().parent / "domains" / "GreenhouseGasPrediction"


def _patched_main_source(template_text: str) -> str:
    patched = template_text.replace(
        'print(f"Train MSE: {train_mse}")',
        'train_mse = float(mse)\n    print(f"Train MSE: {train_mse}")',
    )
    patched = patched.replace(
        'print(json.dumps({"Test MSE": test_mse, "Train MSE": train_mse}))',
        'print(json.dumps({"Test MSE": float(test_mse), "Train MSE": float(train_mse)}))',
    )
    return patched


def _ensure_task_cache() -> Path:
    marker = TASK_CACHE_DIR / ".ready"
    if marker.exists():
        return TASK_CACHE_DIR

    domain_root = _discogen_domain_root()
    template_root = domain_root / "templates" / "default"
    datasets_root = domain_root / "datasets"

    if TASK_CACHE_DIR.exists():
        shutil.rmtree(TASK_CACHE_DIR)
    TASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    main_source = _patched_main_source((template_root / "main.py").read_text())
    model_source = (template_root / "base" / "model.py").read_text()
    data_processing_source = (template_root / "base" / "data_processing.py").read_text()

    for dataset in ALL_DATASETS:
        dataset_dir = TASK_CACHE_DIR / dataset
        data_dir = dataset_dir / "data"
        src_dataset_dir = datasets_root / dataset

        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(next(src_dataset_dir.glob("*_train.npy")), data_dir / "train_data.npy")
        shutil.copy2(next(src_dataset_dir.glob("*_test.npy")), data_dir / "test_data.npy")

        (dataset_dir / "main.py").write_text(main_source)
        (dataset_dir / "model.py").write_text(model_source)
        (dataset_dir / "data_processing.py").write_text(data_processing_source)

    marker.write_text(f"ready:{SLICE}")
    return TASK_CACHE_DIR


def _run_dataset(dataset_dir: Path, timeout: int = DATASET_TIMEOUT_SECONDS) -> dict[str, Any]:
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
    raise RuntimeError("no JSON metrics in stdout")


def _run_dataset_isolated(dataset_dir: Path, code_str: str, timeout: int = DATASET_TIMEOUT_SECONDS) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"levi_ghg_{dataset_dir.name}_") as tmp_dir:
        tmp_dataset_dir = Path(tmp_dir) / dataset_dir.name
        shutil.copytree(dataset_dir, tmp_dataset_dir)
        (tmp_dataset_dir / TARGET_MODULE_FILE).write_text(code_str)
        return _run_dataset(tmp_dataset_dir, timeout=timeout)


def score_fn(target_fn: Any, inputs: list[str]) -> dict[str, Any]:
    code_str = target_fn.__globals__.get("__source_code__")
    if not code_str:
        return {"error": "no source code available (missing __source_code__ in globals)"}

    try:
        task_root = _ensure_task_cache()
    except Exception as e:
        return {"error": str(e)}

    datasets = tuple(inputs) if inputs else TRAIN_DATASETS

    test_mses: dict[str, float] = {}
    train_mses: dict[str, float] = {}
    errors: dict[str, str] = {}
    start = time.perf_counter()

    for dataset in datasets:
        dataset_dir = task_root / dataset
        if not dataset_dir.is_dir():
            errors[dataset] = "dataset directory missing"
            continue

        try:
            metrics = _run_dataset_isolated(dataset_dir, code_str)
            test_mse = float(metrics["Test MSE"])
            train_mse = float(metrics["Train MSE"])
            test_mses[dataset] = test_mse
            train_mses[dataset] = train_mse
        except Exception as e:
            errors[dataset] = str(e)

    elapsed = time.perf_counter() - start
    valid_mses = list(test_mses.values())

    if not valid_mses:
        error_summary = "; ".join(f"{k}: {v}" for k, v in errors.items()) if errors else "unknown error"
        return {"error": f"all datasets failed: {error_summary}"}

    mean_test_mse = sum(valid_mses) / len(valid_mses)
    overall_score = -mean_test_mse

    result: dict[str, Any] = {
        "score": overall_score,
        "mean_test_mse": mean_test_mse,
        "mean_train_mse": sum(train_mses.values()) / len(train_mses),
        "execution_time": elapsed,
        "datasets_ok": len(valid_mses),
        "datasets_total": len(datasets),
        "slice": SLICE,
    }

    for dataset, value in test_mses.items():
        result[f"score_{dataset}"] = -value
        result[f"test_mse_{dataset}"] = test_mses[dataset]
        result[f"train_mse_{dataset}"] = train_mses[dataset]

    if errors:
        result["errors"] = errors

    return result
