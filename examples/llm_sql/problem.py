"""
LLM SQL (CSV Column Reordering) Problem Definition.

Based on ADRS-Leaderboard: https://ucbskyadrs.github.io/leaderboard
Paper: Optimizing LLM Queries in Relational Workloads (arXiv:2403.05821)
"""

from pathlib import Path
import pandas as pd
from utils import evaluate_df_prefix_hit_cnt

# --- Prompts ---

PROBLEM_DESCRIPTION = """
# Column Reordering for Prefix Cache Optimization

**Goal:** Reorder DataFrame columns to maximize prefix hit rate when rows are concatenated into strings.

## How It Works
1. Each row becomes a string by concatenating column values (no spaces)
2. Prefix hit = matching characters from start between consecutive rows
3. Better column order = more prefix sharing = better KV-cache efficiency

## Scoring
```
score = 0.95 * normalized_hit_score + 0.05 * runtime_score
```

## Function Signature
```python
def reorder(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Reorder DataFrame columns to maximize prefix hit rate.

    Args:
        df: Input DataFrame (columns already preprocessed)

    Returns:
        DataFrame with reordered columns (same rows, same columns, different order)
    '''
```

## RULES
1. Return a DataFrame with the SAME rows (same count, can be reordered)
2. Return a DataFrame with the SAME columns (no dropping, no renaming, no adding)
3. Only change: column order and row order
4. Don't use iterrows() or apply(axis=1) - too slow

## Example
```python
def reorder(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Your column reordering logic here
    # Example: sort columns by some score, then sort rows
    cols = list(df.columns)
    # ... reorder cols ...
    df = df[cols]
    df = df.sort_values(by=cols)
    return df
```
"""

FUNCTION_SIGNATURE = """
def reorder(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Reorder DataFrame columns to maximize prefix hit rate.

    Args:
        df: Input DataFrame (up to 30K rows, 60 columns)

    Returns:
        DataFrame with reordered columns and rows (same data, different order)
    '''
    pass
"""

SEED_PROGRAM = '''import pandas as pd
import numpy as np


def reorder(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to maximize prefix sharing."""
    df = df.copy()
    num_rows = len(df)

    # Score each column: prioritize low cardinality and long strings
    col_scores = {}
    for col in df.columns:
        cardinality = df[col].nunique()
        if cardinality == 0 or cardinality == num_rows:
            col_scores[col] = 0
        else:
            avg_len = df[col].astype(str).str.len().mean()
            col_scores[col] = (avg_len ** 2) * (num_rows / cardinality - 1)

    # Reorder columns by score (high to low)
    sorted_cols = sorted(df.columns, key=lambda c: col_scores.get(c, 0), reverse=True)
    df = df[sorted_cols]

    # Sort rows lexicographically
    df = df.sort_values(by=sorted_cols)

    return df
'''

SEED_INSPIRATIONS = []

DIVERSITY_SEED_PROMPT = """
# Column Reordering for Prefix Cache Optimization

Reorder DataFrame columns so consecutive rows share long common prefixes when concatenated.

## Function Signature
```python
def reorder(df: pd.DataFrame) -> pd.DataFrame:
```

## RULES (violations = score 0)
1. Return DataFrame with SAME rows (same count)
2. Return DataFrame with SAME columns (no drop/rename/add)
3. Only change column order and row order
4. No iterrows() or apply(axis=1) - too slow

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

DATASET_SPECS = [
    ("movies.csv", [["movieinfo", "movietitle", "rottentomatoeslink"]], None),
    ("beer.csv", [["beer/beerId", "beer/name"]], None),
    ("BIRD.csv", [["PostId", "Body"]], None),
    ("PDMX.csv", [["path", "metadata"], ["hasmetadata", "isofficial", "isuserpublisher", "isdraft", "hasannotations", "subsetall"]], None),
    ("products.csv", [["product_title", "parent_asin"]], None),
]


def _merge_columns(df, col_merge):
    """Apply column merging - done by evaluator, not LLM."""
    df = df.copy()
    if col_merge:
        for group in col_merge:
            valid = [c for c in group if c in df.columns]
            if len(valid) > 1:
                merged_name = "_".join(valid)
                df[merged_name] = df[valid].astype(str).agg("".join, axis=1)
                df = df.drop(columns=valid)
    return df


def load_datasets(sample_size: int = None):
    """Load all datasets with column merging pre-applied.

    Args:
        sample_size: If provided, sample each dataset to this many rows.
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
        # Pre-merge columns so LLM doesn't have to
        df = _merge_columns(df, col_merge)
        datasets.append((df, filename))
    return datasets


# Full datasets for final evaluation
INPUTS = load_datasets()

# Sampled datasets (1.5K rows each) for quick cascade evaluation
INPUTS_SAMPLED = load_datasets(sample_size=1500)


# --- Baseline Calculation ---

_BASELINE_HIT_RATE = None

def _calculate_baseline_hit_rate(inputs):
    baseline_hit_rates = []
    for df, filename in inputs:
        _, hit_rate = evaluate_df_prefix_hit_cnt(df)
        baseline_hit_rates.append(hit_rate / 100.0)
    return sum(baseline_hit_rates) / len(baseline_hit_rates) if baseline_hit_rates else 0.0


def _get_baseline_hit_rate(inputs):
    global _BASELINE_HIT_RATE
    if _BASELINE_HIT_RATE is None:
        _BASELINE_HIT_RATE = _calculate_baseline_hit_rate(inputs)
    return _BASELINE_HIT_RATE


# --- Score Function ---

def score_fn(reorder_fn, inputs):
    import time
    import warnings
    warnings.filterwarnings("ignore")

    try:
        hit_rates = []
        runtimes = []

        for df, filename in inputs:
            df_copy = df.copy()
            original_row_count = len(df_copy)
            original_cols = set(df_copy.columns)

            start = time.time()
            reordered = reorder_fn(df_copy)
            runtime = time.time() - start
            runtimes.append(runtime)

            # Validate return type
            if not isinstance(reordered, pd.DataFrame):
                return {"error": f"Expected DataFrame, got {type(reordered).__name__}"}

            # Validate row count
            if len(reordered) != original_row_count:
                return {"error": f"Row count mismatch: {len(reordered)} vs {original_row_count}"}

            # Validate columns - must be exactly the same
            result_cols = set(reordered.columns)
            if result_cols != original_cols:
                missing = original_cols - result_cols
                extra = result_cols - original_cols
                if missing:
                    return {"error": f"Missing columns: {missing}"}
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
    except Exception as e:
        return {"error": str(e)}
