"""
LLM SQL (CSV Column Reordering) Problem Definition.

Based on ADRS-Leaderboard: https://ucbskyadrs.github.io/leaderboard
Paper: Optimizing LLM Queries in Relational Workloads (arXiv:2403.05821)

Matches ADRS-Leaderboard evaluator.py:
- Same parameters passed to solve function
- Same col_merge handling
- Same scoring formula
"""

from pathlib import Path
import sys
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from utils import evaluate_df_prefix_hit_cnt

# --- Prompts ---

PROBLEM_DESCRIPTION = """
# LLM SQL Problem

Optimize CSV column ordering to maximize prefix hit rate. Reorder columns so concatenated row values have maximum common prefix overlap between consecutive rows.

**Target**: Maximize prefix hit rate, minimize runtime (up to 10s threshold)

Evaluates on: movies.csv, beer.csv, BIRD.csv, PDMX.csv, products.csv

## API (matches ADRS-Leaderboard parameters)

```python
def solve(
    df: pd.DataFrame,
    early_stop: int = 100000,
    row_stop: int = 4,
    col_stop: int = 2,
    col_merge: list = None,
    one_way_dep: list = None,
    distinct_value_threshold: float = 0.7,
    parallel: bool = True,
) -> pd.DataFrame:
    '''
    Reorder DataFrame columns to maximize prefix hit rate.

    Args:
        df: Input DataFrame (raw, columns NOT pre-merged)
        early_stop: Early stopping threshold
        row_stop: Row stopping threshold
        col_stop: Column stopping threshold
        col_merge: List of column groups to merge, e.g. [["col1", "col2"], ["col3", "col4"]]
        one_way_dep: One-way dependencies (unused, for compatibility)
        distinct_value_threshold: Threshold for distinct values
        parallel: Whether to use parallel processing

    Returns:
        DataFrame with merged columns, reordered columns and rows
    '''
```

The solve() function must:
1. Apply col_merge: merge specified column groups into single columns (concatenate values, drop originals)
2. Reorder columns and rows to maximize prefix hit rate
3. Return the merged + reordered DataFrame

## Column Merging

For col_merge=[["col1", "col2"]], merge into "col1_col2" column:
```python
df["col1_col2"] = df[["col1", "col2"]].apply(lambda x: "".join([f"{val}" for val in x]), axis=1)
df = df.drop(columns=["col1", "col2"])
```

## Scoring

```
baseline_hit_rate = Average prefix hit rate using original column order (after merging)
avg_hit_rate = Your solution's average prefix hit rate
avg_runtime = Average runtime per dataset (seconds)

normalized_hit_score = ((avg_hit_rate - baseline_hit_rate) / (1.0 - baseline_hit_rate)) * 100
normalized_hit_score = clamp(normalized_hit_score, 0, 100)

runtime_component = (10.0 - min(10.0, avg_runtime)) / 10.0 * 100

final_score = 0.95 * normalized_hit_score + 0.05 * runtime_component
```
"""

FUNCTION_SIGNATURE = """
def solve(
    df: pd.DataFrame,
    early_stop: int = 100000,
    row_stop: int = 4,
    col_stop: int = 2,
    col_merge: list = None,
    one_way_dep: list = None,
    distinct_value_threshold: float = 0.7,
    parallel: bool = True,
) -> pd.DataFrame:
    '''
    Reorder DataFrame columns to maximize prefix hit rate.

    Args:
        df: Input DataFrame (raw, columns NOT pre-merged)
        early_stop: Early stopping threshold
        row_stop: Row stopping threshold
        col_stop: Column stopping threshold
        col_merge: List of column groups to merge
        one_way_dep: One-way dependencies (unused)
        distinct_value_threshold: Threshold for distinct values
        parallel: Whether to use parallel processing

    Returns:
        DataFrame with merged columns, reordered columns and rows
    '''
    pass
"""

SEED_PROGRAM = '''import pandas as pd

def solve(
    df: pd.DataFrame,
    early_stop: int = 100000,
    row_stop: int = 4,
    col_stop: int = 2,
    col_merge: list = None,
    one_way_dep: list = None,
    distinct_value_threshold: float = 0.7,
    parallel: bool = True,
) -> pd.DataFrame:
    """Reorder columns to maximize prefix sharing."""
    df = df.copy()

    # Apply column merging (required by evaluator)
    if col_merge:
        for cols_to_merge in col_merge:
            if all(col in df.columns for col in cols_to_merge):
                merged_name = "_".join(cols_to_merge)
                df[merged_name] = df[cols_to_merge].apply(
                    lambda x: "".join([f"{val}" for val in x]), axis=1
                )
                df = df.drop(columns=cols_to_merge)

    # Baseline: sort rows by all columns
    df = df.sort_values(by=list(df.columns))
    return df
'''

SEED_INSPIRATIONS = []

DIVERSITY_SEED_PROMPT = """
# Column Reordering for Prefix Cache Optimization

Reorder DataFrame columns so consecutive rows share long common prefixes when concatenated.

## Function Signature (ADRS-Leaderboard compatible)
```python
def solve(
    df: pd.DataFrame,
    early_stop: int = 100000,
    row_stop: int = 4,
    col_stop: int = 2,
    col_merge: list = None,
    one_way_dep: list = None,
    distinct_value_threshold: float = 0.7,
    parallel: bool = True,
) -> pd.DataFrame:
```

## RULES (violations = score 0)
1. MUST apply col_merge first: merge specified column groups into single columns
2. Return DataFrame with SAME rows (same count)
3. After merging, only change column order and row order
4. No iterrows() or apply(axis=1) on large data - too slow

## Column Merging (REQUIRED)
```python
if col_merge:
    for cols_to_merge in col_merge:
        if all(col in df.columns for col in cols_to_merge):
            merged_name = "_".join(cols_to_merge)
            df[merged_name] = df[cols_to_merge].apply(
                lambda x: "".join([f"{val}" for val in x]), axis=1
            )
            df = df.drop(columns=cols_to_merge)
```

## Your Task
Design a DIFFERENT algorithm than the existing seeds.

## Existing Seeds:
{existing_seeds}

## Output
Output ONLY complete Python code in a ```python block.
"""

META_ADVISOR_PROMPT = """Analyze failures and provide SPECIFIC fixes. Under 100 words.

{metrics_data}

**Fixes:**"""

# --- Dataset Configuration ---

DATASETS_DIR = Path(__file__).parent / "datasets"

# Dataset specs: (filename, col_merge, sample_size)
# col_merge matches ADRS-Leaderboard evaluator.py exactly
DATASET_SPECS = [
    ("movies.csv", [["movieinfo", "movietitle", "rottentomatoeslink"]], None),
    ("beer.csv", [["beer/beerId", "beer/name"]], None),
    ("BIRD.csv", [["PostId", "Body"]], None),
    ("PDMX.csv", [["path", "metadata"], ["hasmetadata", "isofficial", "isuserpublisher", "isdraft", "hasannotations", "subsetall"]], None),
    ("products.csv", [["product_title", "parent_asin"]], None),
]


def _merge_columns(df, col_merge):
    """Apply column merging - matches ADRS-Leaderboard evaluator.py exactly."""
    df = df.copy()
    if col_merge:
        for cols_to_merge in col_merge:
            if all(col in df.columns for col in cols_to_merge):
                merged_name = "_".join(cols_to_merge)
                df[merged_name] = df[cols_to_merge].apply(
                    lambda x: "".join([f"{val}" for val in x]), axis=1
                )
                df = df.drop(columns=cols_to_merge)
    return df


def load_datasets(sample_size: int = None):
    """Load all datasets WITHOUT column merging (raw DataFrames).

    Args:
        sample_size: If provided, sample each dataset to this many rows.

    Returns:
        List of (df, filename, col_merge) tuples - col_merge passed to solver
    """
    datasets = []
    for filename, col_merge, spec_sample_size in DATASET_SPECS:
        path = DATASETS_DIR / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        # Use provided sample_size, fall back to spec_sample_size
        effective_sample = sample_size or spec_sample_size
        if effective_sample and len(df) > effective_sample:
            df = df.sample(effective_sample, random_state=42)
        # Do NOT pre-merge columns - pass raw df + col_merge to solver
        datasets.append((df, filename, col_merge))
    return datasets


class LazyDatasets:
    """Lazily load heavy CSV inputs on first use."""

    def __init__(self, sample_size: int | None = None):
        self._sample_size = sample_size
        self._datasets = None

    def _load(self):
        if self._datasets is None:
            self._datasets = load_datasets(sample_size=self._sample_size)
        return self._datasets

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())

    def __getitem__(self, item):
        return self._load()[item]

    def __repr__(self):
        return repr(self._load())


# Full datasets for final evaluation
INPUTS = LazyDatasets()

# Sampled datasets (1.5K rows each) for quick cascade evaluation
INPUTS_SAMPLED = LazyDatasets(sample_size=1500)


# --- Baseline Calculation ---

_BASELINE_HIT_RATE = None

def _calculate_baseline_hit_rate(inputs):
    """Calculate baseline using original column order after merging.

    Matches ADRS-Leaderboard evaluator.py _process_baseline_dataset().
    """
    baseline_hit_rates = []
    for df, filename, col_merge in inputs:
        # Apply column merges (same as evaluator does for baseline)
        df_merged = _merge_columns(df, col_merge)
        _, hit_rate = evaluate_df_prefix_hit_cnt(df_merged)
        baseline_hit_rates.append(hit_rate / 100.0)
    return sum(baseline_hit_rates) / len(baseline_hit_rates) if baseline_hit_rates else 0.0


def _get_baseline_hit_rate(inputs):
    global _BASELINE_HIT_RATE
    if _BASELINE_HIT_RATE is None:
        _BASELINE_HIT_RATE = _calculate_baseline_hit_rate(inputs)
    return _BASELINE_HIT_RATE


def _get_expected_columns_after_merge(df, col_merge):
    """Get expected column set after applying col_merge."""
    expected_cols = set(df.columns)
    if col_merge:
        for cols_to_merge in col_merge:
            valid_cols = [c for c in cols_to_merge if c in df.columns]
            if len(valid_cols) > 1:
                # Remove original columns, add merged column
                for c in valid_cols:
                    expected_cols.discard(c)
                expected_cols.add("_".join(valid_cols))
    return expected_cols


# --- Score Function ---

def score_fn(solve_fn, inputs):
    """Score function matching ADRS-Leaderboard evaluator.py exactly.

    Args:
        solve_fn: Function with signature solve(df, early_stop, row_stop, col_stop, col_merge, ...)
        inputs: List of (df, filename, col_merge) tuples
    """
    import time
    import warnings
    warnings.filterwarnings("ignore")

    try:
        hit_rates = []
        runtimes = []

        for df, filename, col_merge in inputs:
            df_copy = df.copy()
            original_row_count = len(df_copy)

            # Expected columns after merging
            expected_cols = _get_expected_columns_after_merge(df_copy, col_merge)

            # Call solve() with all parameters - matches ADRS-Leaderboard exactly
            start = time.time()
            reordered = solve_fn(
                df_copy,
                early_stop=100000,
                row_stop=4,
                col_stop=2,
                col_merge=col_merge,
                one_way_dep=[],
                distinct_value_threshold=0.7,
                parallel=True,
            )
            runtime = time.time() - start
            runtimes.append(runtime)

            # Validate return type
            if not isinstance(reordered, pd.DataFrame):
                return {"error": f"Expected DataFrame, got {type(reordered).__name__}"}

            # Validate row count
            if len(reordered) != original_row_count:
                return {"error": f"Row count mismatch: {len(reordered)} vs {original_row_count}"}

            # Validate columns - must match expected after merging
            result_cols = set(reordered.columns)
            if result_cols != expected_cols:
                missing = expected_cols - result_cols
                extra = result_cols - expected_cols
                if missing:
                    return {"error": f"Missing columns after merge: {missing}"}
                if extra:
                    return {"error": f"Extra columns: {extra}"}

            _, hit_rate = evaluate_df_prefix_hit_cnt(reordered)
            hit_rates.append(hit_rate / 100.0)

        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        avg_runtime = sum(runtimes) / len(runtimes)
        baseline_hit_rate = _get_baseline_hit_rate(inputs)

        if baseline_hit_rate >= 1.0:
            normalized_hit_score = 100.0 if avg_hit_rate >= 1.0 else 0.0
        else:
            normalized_hit_score = ((avg_hit_rate - baseline_hit_rate) / (1.0 - baseline_hit_rate)) * 100
            normalized_hit_score = max(0, min(100, normalized_hit_score))

        runtime_component = (10.0 - min(10.0, avg_runtime)) / 10.0 * 100
        score = 0.95 * normalized_hit_score + 0.05 * runtime_component

        return {
            "score": score,
            "hit_rate": avg_hit_rate * 100,
            "normalized_hit_score": normalized_hit_score,
            "baseline_hit_rate": baseline_hit_rate * 100,
            "runtime": avg_runtime,
        }
    except MemoryError:
        return {"error": "MemoryError: code used too much memory"}
    except Exception as e:
        return {"error": str(e) or type(e).__name__}
