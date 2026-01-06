"""
LLM SQL (CSV Column Reordering) Problem Definition.

Contains problem description, prompts, scoring function, and test inputs.
"""

from pathlib import Path

import pandas as pd

from utils import evaluate_df_prefix_hit_cnt

# --- Prompts ---

PROBLEM_DESCRIPTION = """
# CSV Column Reordering Optimization

## Problem
Reorder DataFrame columns to maximize prefix hit rate for LLM prompt caching.

## Key Concepts
- Rows are concatenated into strings: col1_val + col2_val + ...
- Prefix hit = matching characters from start between consecutive rows
- Hit score = sum of squared lengths of matching prefix fields
- Higher prefix reuse = better caching efficiency

## Objective
Find optimal column ordering to maximize prefix hits across all rows.
- Baseline (original order) gives ~7% hit rate
- Better orderings group similar values together

## Input
- `df`: pandas DataFrame to reorder
- `col_merge`: Column groups to merge, e.g., [["col1", "col2"]]
- `row_stop`, `col_stop`: Recursion depth limits
- `distinct_value_threshold`: Filter high-cardinality columns

## Evaluation
Your function is called on 5 datasets. Score combines hit rate (95%) and runtime (5%).

## You can import standard library modules (pandas, numpy, collections, random, math, concurrent.futures, etc.)
"""

FUNCTION_SIGNATURE = """
def reorder(
    df: pd.DataFrame,
    early_stop: int = 0,
    row_stop: int = None,
    col_stop: int = None,
    col_merge: list = None,
    one_way_dep: list = None,
    distinct_value_threshold: float = 0.7,
    parallel: bool = True,
) -> pd.DataFrame:
    '''
    Reorder DataFrame columns to maximize prefix hit rate.

    Args:
        df: Input DataFrame
        col_merge: Column groups to merge before reordering
        row_stop, col_stop: Recursion depth limits
        distinct_value_threshold: Filter columns with >threshold unique ratio

    Returns:
        Reordered DataFrame with same rows, optimized column order
    '''
    pass
"""

SEED_PROGRAM = '''import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from collections import Counter
import numpy as np


@lru_cache(maxsize=None)
def calculate_length(value):
    """Calculate squared length for scoring."""
    if isinstance(value, str):
        return len(value) ** 2
    if isinstance(value, (int, float)):
        return len(str(value)) ** 2
    return 16


def merging_columns(df: pd.DataFrame, col_names: list[str]) -> pd.DataFrame:
    """Merge multiple columns into one."""
    existing_cols = [c for c in col_names if c in df.columns]
    if len(existing_cols) < 2:
        return df
    merged_name = "_".join(existing_cols)
    df[merged_name] = df[existing_cols].apply(lambda x: "".join([str(val) for val in x]), axis=1)
    return df.drop(columns=existing_cols)


class Reorderer:
    """GGR (Greedy Group Reordering) algorithm for maximizing prefix hit rate."""

    def __init__(self):
        self.val_len = {}
        self.base = 5000
        self.row_stop = None
        self.col_stop = None

    def find_max_group_value(self, value_counts: dict, early_stop: int = 0):
        """Find the value with maximum weighted count (length^2 * (count-1))."""
        best_val, best_score = None, -1
        for val, count in value_counts.items():
            if count <= 1:
                continue
            vl = self.val_len.get(val, calculate_length(val))
            self.val_len[val] = vl
            score = vl * (count - 1)
            if score > best_score:
                best_score, best_val = score, val
        return best_val if best_score >= early_stop else None

    def column_recursion(self, max_value, grouped_rows, row_stop, col_stop, early_stop):
        values = grouped_rows.values
        mask = (values == max_value)
        idx = np.argsort(~mask, axis=1, kind='stable')
        sorted_values = values[np.arange(len(grouped_rows))[:, None], idx]
        remainder_values = sorted_values[:, 1:]
        remainder_counts = Counter(remainder_values.ravel())
        reordered_remainder = self.recursive_reorder(
            pd.DataFrame(remainder_values), remainder_counts, early_stop, row_stop, col_stop + 1
        )
        result_df = pd.DataFrame(np.hstack([
            np.full((len(grouped_rows), 1), max_value, dtype=object),
            reordered_remainder.values
        ]))
        return result_df, remainder_counts

    def fixed_reorder(self, df: pd.DataFrame, row_sort: bool = True) -> pd.DataFrame:
        cols = [c for c in df.columns if c != "original_index"]
        if not cols:
            return df
        sample = df.sample(min(len(df), 1000), random_state=42) if len(df) > 1000 else df
        scores = {}
        for col in cols:
            try:
                vc = sample[col].value_counts(normalize=True)
                p = (vc ** 2).sum()
                avg_sq_len = (sample[col].astype(str).str.len() ** 2).mean()
                scores[col] = avg_sq_len * p / (1 - p + 1e-9) if not pd.isna(avg_sq_len) else 0
            except:
                scores[col] = 0
        sorted_cols = sorted(cols, key=lambda x: scores.get(x, 0), reverse=True)
        if "original_index" in df.columns:
            sorted_cols.append("original_index")
        reordered_df = df[sorted_cols]
        if row_sort:
            sort_cols = [c for c in sorted_cols if c != "original_index"]
            if sort_cols:
                reordered_df = reordered_df.sort_values(by=sort_cols)
        return reordered_df

    def recursive_reorder(self, df: pd.DataFrame, value_counts: dict, early_stop: int = 0, row_stop: int = 0, col_stop: int = 0) -> pd.DataFrame:
        if df.empty or df.shape[1] <= 1:
            return self.fixed_reorder(df)
        if (self.row_stop and row_stop >= self.row_stop) or (self.col_stop and col_stop >= self.col_stop):
            return self.fixed_reorder(df)
        max_value = self.find_max_group_value(value_counts, early_stop)
        if max_value is None:
            return self.fixed_reorder(df)
        mask = (df.values == max_value).any(axis=1)
        if not mask.any():
            return self.fixed_reorder(df)
        grouped_rows = df[mask]
        remaining_rows = df[~mask]
        reordered_grouped, grouped_counts = self.column_recursion(max_value, grouped_rows, row_stop, col_stop, early_stop)
        if not remaining_rows.empty:
            remaining_counts = value_counts.copy()
            remaining_counts[max_value] -= len(grouped_rows)
            if remaining_counts[max_value] <= 0:
                del remaining_counts[max_value]
            for k, v in grouped_counts.items():
                if k in remaining_counts:
                    remaining_counts[k] -= v
                    if remaining_counts[k] <= 0:
                        del remaining_counts[k]
            reordered_remaining = self.recursive_reorder(remaining_rows, remaining_counts, early_stop, row_stop + 1, col_stop)
            return pd.DataFrame(np.vstack([reordered_grouped.values, reordered_remaining.values]))
        return reordered_grouped

    def recursive_split_and_reorder(self, df: pd.DataFrame, early_stop: int = 0):
        if len(df) <= self.base:
            return self.recursive_reorder(df, Counter(df.values.ravel()), early_stop)
        mid = len(df) // 2
        with ThreadPoolExecutor(max_workers=2) as exc:
            f1 = exc.submit(self.recursive_split_and_reorder, df.iloc[:mid], early_stop)
            f2 = exc.submit(self.recursive_split_and_reorder, df.iloc[mid:], early_stop)
            return pd.concat([f1.result(), f2.result()], axis=0, ignore_index=True)


def reorder(
    df: pd.DataFrame,
    early_stop: int = 0,
    row_stop: int = None,
    col_stop: int = None,
    col_merge: list = None,
    one_way_dep: list = None,
    distinct_value_threshold: float = 0.7,
    parallel: bool = True,
) -> pd.DataFrame:
    """Main entry point for column reordering."""
    df = df.copy()

    # Apply column merges
    if col_merge:
        for group in col_merge:
            df = merging_columns(df, group)

    reorderer = Reorderer()
    reorderer.row_stop = row_stop if row_stop else len(df)
    reorderer.col_stop = col_stop if col_stop else len(df.columns)

    # Add tracking index
    df["original_index"] = range(len(df))

    # Filter high-cardinality columns
    threshold = len(df) * distinct_value_threshold
    cols_to_keep = [c for c in df.columns if c == "original_index" or df[c].nunique() <= threshold]
    cols_to_discard = [c for c in df.columns if c != "original_index" and df[c].nunique() > threshold]

    df_recurse = df[cols_to_keep]
    df_discard = df[["original_index"] + cols_to_discard] if cols_to_discard else None

    # Pre-compute value lengths
    for v in pd.unique(df_recurse.values.ravel()):
        reorderer.val_len[v] = calculate_length(v)

    # Reorder
    if parallel:
        reordered = reorderer.recursive_split_and_reorder(df_recurse, early_stop)
    else:
        reordered = reorderer.recursive_reorder(df_recurse, Counter(df_recurse.values.ravel()), early_stop)

    reordered.columns = list(range(len(reordered.columns) - 1)) + ["original_index"]

    # Merge back discarded columns
    if df_discard is not None:
        reordered = pd.merge(reordered, df_discard, on="original_index", how="left")

    reordered = reordered.drop(columns=["original_index"])
    return reordered.sort_values(by=list(reordered.columns))
'''

SEED_INSPIRATIONS = []

DIVERSITY_SEED_PROMPT = """
# CSV Column Reordering Optimization

## Problem
Reorder DataFrame columns to maximize prefix hit rate for LLM prompt caching.

## Key Concepts
- Rows become strings: col1_val + col2_val + ...
- Prefix hit = matching chars from start between consecutive rows
- Hit score = sum of squared lengths of matching prefix fields

## Input
- `df`: pandas DataFrame with text data
- `col_merge`: Column groups to merge
- `row_stop`, `col_stop`: Recursion limits

## Function Signature
```python
def reorder(
    df: pd.DataFrame,
    early_stop: int = 0,
    row_stop: int = None,
    col_stop: int = None,
    col_merge: list = None,
    one_way_dep: list = None,
    distinct_value_threshold: float = 0.7,
    parallel: bool = True,
) -> pd.DataFrame:
    '''Returns reordered DataFrame with optimized column order.'''
    pass
```

## You can import standard library modules (pandas, numpy, collections, random, math, concurrent.futures, etc.)

## Your Task: ALGORITHMIC DIVERSITY

Design a solution using a **FUNDAMENTALLY DIFFERENT ALGORITHM** than the existing seeds.

**DO NOT:**
- Make minor variations or parameter tweaks
- Use the same core algorithm with different constants

**DO:**
- Analyze what paradigm each existing seed uses
- Design from first principles using a different strategy
- Consider what information the existing seeds are NOT using

## Existing Seeds:
{existing_seeds}

## Output
Output ONLY the complete Python code in a ```python block.
"""

META_ADVISOR_PROMPT = """You are a lessons-learned advisor for an evolutionary code optimization system.

## Your Role
Analyze FAILURES from recent evaluations. Your lessons get injected into LLM prompts to help future solutions avoid the same mistakes.

## What You're Given
- **Failure count**: How many candidates failed (crashes, invalid code, timeouts, etc.)
- **Error patterns**: Specific error messages encountered (including timeouts)
- **Previous lessons**: What you warned about last time

## Your Task: Write Concise Lessons (150-200 words max)

### Focus ONLY on Failure Prevention
You do NOT see successful solutions. Your job is purely defensive:
1. **Identify error patterns** - What mistakes are being made repeatedly?
2. **Explain root causes** - Why are these errors happening?
3. **Give specific fixes** - Exactly how to avoid each error type

### For Each Error Pattern:
- Quote the error briefly
- Explain what causes it
- Give a specific fix

## Output Format
Keep it SHORT and DIRECT:

**Avoid These Errors:**
- [Error pattern]: [How to fix]
- [Error pattern]: [How to fix]

---

{metrics_data}

Provide your lessons (150-200 words max):"""

# --- Dataset Configuration ---

DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "ADRS/openevolve/examples/ADRS/llm_sql/datasets"

# Dataset specs: (filename, col_merge, sample_size or None for full)
DATASET_SPECS = [
    ("movies.csv", [["movieinfo", "movietitle", "rottentomatoeslink"]], None),
    ("beer.csv", [["beer/beerId", "beer/name"]], None),
    ("BIRD.csv", [["PostId", "Body"]], 3000),  # Sample to 3K rows for speed
    ("PDMX.csv", [["path", "metadata"], ["hasmetadata", "isofficial", "isuserpublisher", "isdraft", "hasannotations", "subsetall"]], None),
    ("products.csv", [["product_title", "parent_asin"]], None),
]


def load_datasets():
    """Load all datasets with optional sampling."""
    datasets = []
    for filename, col_merge, sample_size in DATASET_SPECS:
        path = DATASETS_DIR / filename
        if not path.exists():
            print(f"Warning: Dataset not found: {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
        datasets.append((df, col_merge, filename))
        print(f"  {filename}: {len(df)} rows, {len(df.columns)} cols")
    return datasets


def calculate_baseline(datasets):
    """Calculate baseline hit rate using original column order."""
    hit_rates = []
    for df, col_merge, filename in datasets:
        df_eval = df.copy()
        if col_merge:
            for group in col_merge:
                valid = [c for c in group if c in df_eval.columns]
                if len(valid) > 1:
                    merged_name = "_".join(valid)
                    df_eval[merged_name] = df_eval[valid].apply(
                        lambda x: "".join(str(v) for v in x), axis=1
                    )
                    df_eval = df_eval.drop(columns=valid)
        _, hit_rate = evaluate_df_prefix_hit_cnt(df_eval)
        hit_rates.append(hit_rate / 100.0)
    return sum(hit_rates) / len(hit_rates) if hit_rates else 0.0


# --- Load Data ---
print("Loading datasets...")
INPUTS = load_datasets()
print("Calculating baseline...")
BASELINE = calculate_baseline(INPUTS)


# --- Score Function ---

def score_fn(reorder_fn, inputs):
    """Evaluate reordering function: returns 0-100 score based on prefix hit improvement."""
    import time

    try:
        hit_rates = []
        runtimes = []

        for df, col_merge, filename in inputs:
            df_copy = df.copy()

            start = time.time()
            reordered = reorder_fn(
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

            if not isinstance(reordered, pd.DataFrame):
                return {"error": f"Expected DataFrame, got {type(reordered).__name__}"}
            if len(reordered) != len(df):
                return {"error": f"Row count mismatch: {len(reordered)} vs {len(df)}"}

            _, hit_rate = evaluate_df_prefix_hit_cnt(reordered)
            hit_rates.append(hit_rate / 100.0)

        avg_hit = sum(hit_rates) / len(hit_rates)
        avg_runtime = sum(runtimes) / len(runtimes)

        # Normalized hit score (0-100)
        if BASELINE >= 1.0:
            normalized_hit_score = 100.0 if avg_hit >= 1.0 else 0.0
        else:
            normalized_hit_score = ((avg_hit - BASELINE) / (1.0 - BASELINE)) * 100
            normalized_hit_score = max(0.0, min(100.0, normalized_hit_score))

        # Runtime component (100 at 0s, 0 at 10s)
        runtime_component = (10.0 - min(10.0, avg_runtime)) / 10.0 * 100

        # Combined score: 95% hit rate, 5% runtime
        score = 0.95 * normalized_hit_score + 0.05 * runtime_component

        return {"score": score, "hit_rate": avg_hit * 100, "runtime": avg_runtime}
    except Exception as e:
        return {"error": str(e)}
