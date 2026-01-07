"""
LLM SQL (CSV Column Reordering) Problem Definition.

Contains problem description, prompts, scoring function, and test inputs.
"""

from pathlib import Path

import pandas as pd

from utils import evaluate_df_prefix_hit_cnt

# --- Prompts ---

PROBLEM_DESCRIPTION = """
# LLM SQL Problem

Optimize CSV column ordering to maximize prefix hit rate. Reorder columns so concatenated row values have maximum common prefix overlap between consecutive rows.

## Key Concepts
- Rows are concatenated into strings (no spaces): col1_val + col2_val + ...
- Prefix hit = matching characters from start between consecutive rows
- Higher prefix reuse = better caching efficiency

## Input
- `df`: pandas DataFrame to reorder (up to 30K rows, up to 60 columns)
- `col_merge`: Column groups to merge, e.g., [["col1", "col2"]]
- `row_stop`, `col_stop`: Recursion depth limits
- `distinct_value_threshold`: Filter high-cardinality columns

## Scoring

```
baseline_hit_rate = Average prefix hit rate using original column order
avg_hit_rate = Your solution's average prefix hit rate
avg_runtime = Average runtime per dataset (seconds)

normalized_hit_score = ((avg_hit_rate - baseline_hit_rate) / (1.0 - baseline_hit_rate)) × 100
normalized_hit_score = clamp(normalized_hit_score, 0, 100)

runtime_component = (10.0 - min(10.0, avg_runtime)) / 10.0 × 100

final_score = 0.95 × normalized_hit_score + 0.05 × runtime_component
```

Runtime component: 100 at 0s, 0 at 10s. Solutions ≥10s get 0 runtime component.

## Requirements
- Return a valid pd.DataFrame with the same number of rows as input
- Use only thread-based parallelism (no process-based parallelism)
- Do not use print statements or logging
- Handle edge cases: empty DataFrames, single-column DataFrames, etc.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import Counter


def calculate_length(value):
    if isinstance(value, bool):
        return 4 ** 2
    if isinstance(value, (int, float)):
        return len(str(value)) ** 2
    if isinstance(value, str):
        return len(value) ** 2
    return 0


def calculate_col_stats(df, enable_index=False):
    num_rows = len(df)
    column_stats = []
    for col in df.columns:
        if col == "original_index":
            continue
        num_groups = df[col].nunique()
        if df[col].dtype == "object" or df[col].dtype == "string":
            avg_length = df[col].astype(str).str.len().mean()
        elif df[col].dtype == "bool":
            avg_length = 4
        elif df[col].dtype in ["int64", "float64"]:
            avg_length = df[col].astype(str).str.len().mean()
        else:
            avg_length = 0
        avg_length = avg_length ** 2
        if num_groups == 0:
            score = 0
        else:
            avg_size_per_group = num_rows / num_groups
            score = avg_length * (avg_size_per_group - 1)
            if num_rows == num_groups:
                score = 0
        column_stats.append((col, num_groups, avg_length, score))
    if enable_index and "original_index" in df.columns:
        column_stats.append(("original_index", len(df), 0, 0))
    column_stats.sort(key=lambda x: x[3], reverse=True)
    return num_rows, column_stats


def merging_columns(df, col_names):
    if not all(col in df.columns for col in col_names):
        return df
    merged_names = "_".join(col_names)
    df[merged_names] = df[col_names].apply(lambda x: "".join([f"{val}" for val in x]), axis=1)
    df = df.drop(columns=col_names)
    return df


class Reorderer:
    def __init__(self):
        self.num_rows = 0
        self.num_cols = 0
        self.column_stats = None
        self.val_len = None
        self.row_stop = None
        self.col_stop = None
        self.base = 2000

    def find_max_group_value(self, df, value_counts, early_stop=0):
        value_counts = Counter(df.stack())
        weighted_counts = {val: self.val_len[val] * (count - 1) for val, count in value_counts.items()}
        if not weighted_counts:
            return None
        max_group_val, max_weighted_count = max(weighted_counts.items(), key=lambda x: x[1])
        if max_weighted_count < early_stop:
            return None
        return max_group_val

    def reorder_columns_for_value(self, row, value, column_names, grouped_rows_len=1):
        cols_with_value = []
        for idx, col in enumerate(column_names):
            if hasattr(row, col) and getattr(row, col) == value:
                cols_with_value.append(col)
            elif hasattr(row, col.replace(" ", "_")) and getattr(row, col.replace(" ", "_")) == value:
                cols_with_value.append(col)
            else:
                attr_name = f"_{idx}"
                if hasattr(row, attr_name) and getattr(row, attr_name) == value:
                    cols_with_value.append(attr_name)
        cols_without_value = []
        for idx, col in enumerate(column_names):
            if hasattr(row, col) and getattr(row, col) != value:
                cols_without_value.append(col)
            elif hasattr(row, col.replace(" ", "_")) and getattr(row, col.replace(" ", "_")) != value:
                cols_without_value.append(col)
            else:
                attr_name = f"_{idx}"
                if hasattr(row, attr_name) and getattr(row, attr_name) != value:
                    cols_without_value.append(attr_name)
        reordered_cols = cols_with_value + cols_without_value
        return [getattr(row, col) for col in reordered_cols], cols_with_value

    def fixed_reorder(self, df, row_sort=True):
        num_rows, column_stats = calculate_col_stats(df, enable_index=True)
        reordered_columns = [col for col, _, _, _ in column_stats]
        reordered_df = df[reordered_columns]
        if row_sort:
            reordered_df = reordered_df.sort_values(by=reordered_columns, axis=0)
        return reordered_df

    def column_recursion(self, result_df, max_value, grouped_rows, row_stop, col_stop, early_stop):
        cols_settled = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.reorder_columns_for_value, row, max_value, grouped_rows.columns.tolist(), len(grouped_rows))
                for row in grouped_rows.itertuples(index=False)
            ]
            for i, future in enumerate(as_completed(futures)):
                reordered_row, cols_settled = future.result()
                result_df.loc[i] = reordered_row
        grouped_value_counts = Counter()
        if not result_df.empty:
            grouped_result_df = result_df.groupby(result_df.columns[0])
            grouped_value_counts = Counter(grouped_rows.stack())
            for _, group in grouped_result_df:
                if group[group.columns[0]].iloc[0] != max_value:
                    continue
                group_remainder = group.iloc[:, 1:]
                grouped_remainder_value_counts = Counter(group_remainder.stack())
                reordered_group_remainder = self.recursive_reorder(
                    group_remainder, grouped_remainder_value_counts, early_stop=early_stop, row_stop=row_stop, col_stop=col_stop + 1
                )
                group.iloc[:, 1:] = reordered_group_remainder.values
                result_df.update(group)
                break
        return result_df, grouped_value_counts

    def recursive_reorder(self, df, value_counts, early_stop=0, row_stop=0, col_stop=0):
        if df.empty or len(df.columns) == 0 or len(df) == 0:
            return df
        if self.row_stop is not None and row_stop >= self.row_stop:
            return self.fixed_reorder(df)
        if self.col_stop is not None and col_stop >= self.col_stop:
            return self.fixed_reorder(df)
        max_value = self.find_max_group_value(df, value_counts, early_stop=early_stop)
        if max_value is None:
            return self.fixed_reorder(df)
        mask = (df.values == max_value).any(axis=1)
        if not mask.any():
            return self.fixed_reorder(df)
        grouped_rows = df[mask]
        remaining_rows = df[~mask]
        result_df = pd.DataFrame(columns=df.columns)
        reordered_grouped, grouped_counts = self.column_recursion(result_df, max_value, grouped_rows, row_stop, col_stop, early_stop)
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
            return pd.concat([reordered_grouped, reordered_remaining], axis=0, ignore_index=True)
        return reordered_grouped

    def recursive_split_and_reorder(self, df, early_stop=0):
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
    df = df.copy()
    reorderer = Reorderer()

    # Apply column merges (use col_stats order like ADRS)
    if col_merge:
        _, column_stats = calculate_col_stats(df, enable_index=True)
        reordered_columns = [col for col, _, _, _ in column_stats]
        for col_to_merge in col_merge:
            final_col_order = [col for col in reordered_columns if col in col_to_merge]
            if len(final_col_order) > 1:
                df = merging_columns(df, final_col_order)
    reorderer.row_stop = row_stop if row_stop else len(df)
    reorderer.col_stop = col_stop if col_stop else len(df.columns)

    # Calculate column stats
    reorderer.num_rows, column_stats_list = calculate_col_stats(df, enable_index=True)
    reorderer.column_stats = {col: (num_groups, avg_len, score) for col, num_groups, avg_len, score in column_stats_list}

    # Add tracking index
    df["original_index"] = range(len(df))

    # Filter high-cardinality columns
    threshold = len(df) * distinct_value_threshold
    cols_to_keep = [c for c in df.columns if c == "original_index" or df[c].nunique() <= threshold]
    cols_to_discard = [c for c in df.columns if c != "original_index" and df[c].nunique() > threshold]
    cols_to_discard = sorted(cols_to_discard, key=lambda x: reorderer.column_stats.get(x, (0, 0, 0))[2], reverse=True)

    df_recurse = df[cols_to_keep]
    df_discard = df[["original_index"] + cols_to_discard] if cols_to_discard else None

    # Update column_stats to exclude discarded columns
    reorderer.column_stats = {col: stats for col, stats in reorderer.column_stats.items() if col not in cols_to_discard}

    # Pre-compute value lengths
    initial_value_counts = Counter(df_recurse.values.ravel())
    reorderer.val_len = {val: calculate_length(val) for val in initial_value_counts.keys()}

    # Initial fixed reorder (matches ADRS)
    df_recurse = reorderer.fixed_reorder(df_recurse)

    # Save column order before recursive processing (excluding original_index)
    kept_col_names = [c for c in df_recurse.columns if c != "original_index"]

    # Reorder
    if parallel:
        reordered = reorderer.recursive_split_and_reorder(df_recurse, early_stop)
    else:
        reordered = reorderer.recursive_reorder(df_recurse, initial_value_counts, early_stop)

    # Restore original column names (algorithm uses integer indices internally)
    reordered.columns = kept_col_names + ["original_index"]

    # Merge back discarded columns
    if df_discard is not None:
        reordered = pd.merge(reordered, df_discard, on="original_index", how="left")

    reordered = reordered.drop(columns=["original_index"])
    return reordered.sort_values(by=list(reordered.columns))
'''

SEED_INSPIRATIONS = []

DIVERSITY_SEED_PROMPT = """
# LLM SQL Problem

Optimize CSV column ordering to maximize prefix hit rate. Reorder columns so concatenated row values have maximum common prefix overlap between consecutive rows.

## Key Concepts
- Rows become strings (no spaces): col1_val + col2_val + ...
- Prefix hit = matching chars from start between consecutive rows
- Higher prefix reuse = better caching efficiency

## Scoring
```
normalized_hit_score = ((avg_hit_rate - baseline_hit_rate) / (1.0 - baseline_hit_rate)) × 100
runtime_component = (10.0 - min(10.0, avg_runtime)) / 10.0 × 100
final_score = 0.95 × normalized_hit_score + 0.05 × runtime_component
```

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

## Requirements
- Return a valid pd.DataFrame with the same number of rows as input
- Use only thread-based parallelism (no process-based parallelism)
- Do not use print statements or logging

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

DATASETS_DIR = Path(__file__).parent / "datasets"

# Dataset specs: (filename, col_merge, sample_size or None for full)
DATASET_SPECS = [
    ("movies.csv", [["movieinfo", "movietitle", "rottentomatoeslink"]], None),
    ("beer.csv", [["beer/beerId", "beer/name"]], None),
    ("BIRD.csv", [["PostId", "Body"]], None),
    ("PDMX.csv", [["path", "metadata"], ["hasmetadata", "isofficial", "isuserpublisher", "isdraft", "hasannotations", "subsetall"]], None),
    ("products.csv", [["product_title", "parent_asin"]], None),
]


def load_datasets():
    """Load all datasets with optional sampling."""
    datasets = []
    for filename, col_merge, sample_size in DATASET_SPECS:
        path = DATASETS_DIR / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
        datasets.append((df, col_merge, filename))
    return datasets


# --- Load Data ---
INPUTS = load_datasets()


# --- Baseline Calculation ---

_BASELINE_HIT_RATE = None

def _calculate_baseline_hit_rate(inputs):
    baseline_hit_rates = []
    for df, col_merge, filename in inputs:
        df_copy = df.copy()
        if col_merge:
            for col_to_merge in col_merge:
                if all(col in df_copy.columns for col in col_to_merge):
                    merged_name = "_".join(col_to_merge)
                    df_copy[merged_name] = df_copy[col_to_merge].apply(
                        lambda x: "".join([f"{val}" for val in x]), axis=1
                    )
                    df_copy = df_copy.drop(columns=col_to_merge)
        _, hit_rate = evaluate_df_prefix_hit_cnt(df_copy)
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

            # Validate columns: structure-based (allows any merged column naming)
            original_cols = set(df.columns)
            merged_away = set()
            num_merge_groups = 0
            for group in (col_merge or []):
                if all(c in original_cols for c in group):
                    merged_away.update(group)
                    num_merge_groups += 1

            non_merged = original_cols - merged_away
            result_cols = set(reordered.columns)

            # 1. Non-merged columns must be present
            missing_non_merged = non_merged - result_cols
            if missing_non_merged:
                return {"error": f"Missing columns: {missing_non_merged}"}

            # 2. Merged columns must be absent (they should be merged)
            present_merged = merged_away & result_cols
            if present_merged:
                return {"error": f"Columns should have been merged: {present_merged}"}

            # 3. Column count must be correct (non-merged + one per merge group)
            expected_count = len(non_merged) + num_merge_groups
            if len(result_cols) != expected_count:
                return {"error": f"Column count mismatch: expected {expected_count}, got {len(result_cols)}"}

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
