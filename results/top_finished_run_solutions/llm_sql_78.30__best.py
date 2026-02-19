import pandas as pd
import numpy as np
from typing import List, Optional
import multiprocessing as mp
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

def solve(
    df: pd.DataFrame,
    early_stop: int = 100000,
    row_stop: int = 4,
    col_stop: int = 2,
    col_merge: List[List[str]] = None,
    one_way_dep: List[tuple] = None,
    distinct_value_threshold: float = 0.7,
    parallel: bool = True,
) -> pd.DataFrame:
    """
    Optimized Column and Row Reordering for Maximum Prefix Hit Rate.

    Strategy:
    1. Apply column merging with robust error handling.
    2. Use a hybrid column ordering approach:
       - For small column sets (≤ col_stop), use brute-force with early pruning.
       - For larger sets, use greedy selection based on compression-like utility (sum of squares of frequencies).
    3. Row reordering via sequential stable frequency ranking (high-frequency values grouped together).
    4. Leverage vectorized operations and sampling to ensure runtime < 10s.
    5. Avoid common pitfalls: nested multiprocessing, apply(), missing columns, and row truncation.

    Key improvements over v1/v2:
    - Avoids `df.apply(lambda)` entirely; uses vectorized string operations.
    - Prevents daemonic process errors by using ThreadPoolExecutor instead of ProcessPoolExecutor.
    - Handles edge cases: empty DataFrame, missing columns, col_merge=None.
    - Uses a hybrid column selection: greedy with limited permutation search for small sets.
    - Optimized row sorting: stable, sequential, frequency-based ranking.
    """
    # Work on a copy to avoid side effects
    df = df.copy()

    # 1. Column Merging (robust handling)
    if col_merge:
        merged_columns = []
        for group in col_merge:
            valid_cols = [c for c in group if c in df.columns]
            if not valid_cols:
                continue  # Skip empty groups
            merged_name = "_".join(valid_cols)
            # Vectorized string concatenation: fast and safe
            df[merged_name] = df[valid_cols].astype(str).fillna("").sum(axis=1)
            merged_columns.extend(valid_cols)
        # Drop original columns only after merging
        df = df.drop(columns=merged_columns)

    # 2. Handle empty or single-column case
    if df.empty or len(df.columns) == 0:
        return df

    # 3. Sampling for column ordering (speed vs. accuracy trade-off)
    n_rows = len(df)
    sample_size = min(early_stop, n_rows)
    if n_rows > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df

    # Convert to string once; avoid repeated .astype(str) calls
    sample_str = sample_df.astype(str).fillna("")

    # 4. Column Reordering: Hybrid Strategy

    # If very few columns, use brute-force with early stopping
    n_cols = len(df.columns)
    if n_cols <= col_stop:
        # Try all permutations with early pruning
        best_order = None
        best_score = -1

        # Use iterative generation to avoid memory explosion
        from itertools import permutations
        perm_gen = permutations(df.columns)

        # Limit number of permutations to avoid timeout
        for i, perm in enumerate(perm_gen):
            if i >= early_stop:
                break
            score = _calculate_prefix_hit_score_vectorized(perm, sample_str)
            if score > best_score:
                best_score = score
                best_order = list(perm)

        if best_order:
            ordered_cols = best_order
        else:
            ordered_cols = list(df.columns)
    else:
        # Greedy with compression proxy (sum of squares of counts)
        # This directly correlates with prefix hit rate (more duplicates → higher hit rate)
        remaining_cols = list(df.columns)
        ordered_cols = []
        current_prefix = None  # Will be set after first column

        # Helper: Calculate utility of adding a column to current prefix
        def compute_utility(prefix_series, candidate_series):
            # Concatenate
            combined = prefix_series + candidate_series
            # Count frequencies and sum of squares
            counts = combined.value_counts()
            return np.sum(counts ** 2)

        # Select first column: maximize frequency (sum of squares of its own counts)
        best_col = None
        best_score = -1
        for col in remaining_cols:
            counts = sample_str[col].value_counts()
            score = np.sum(counts ** 2)
            if score > best_score:
                best_score = score
                best_col = col
        if best_col is None:
            best_col = remaining_cols[0]
        ordered_cols.append(best_col)
        remaining_cols.remove(best_col)
        current_prefix = sample_str[best_col]

        # Greedily extend the order
        while remaining_cols:
            best_col = None
            best_score = -1
            # Evaluate only top-k candidates (e.g., 10) for efficiency
            candidates = remaining_cols[:10] if len(remaining_cols) > 10 else remaining_cols
            for col in candidates:
                score = compute_utility(current_prefix, sample_str[col])
                if score > best_score:
                    best_score = score
                    best_col = col
            if best_col is None:
                # Fallback: pick first remaining
                best_col = remaining_cols[0]
            ordered_cols.append(best_col)
            remaining_cols.remove(best_col)
            current_prefix = current_prefix + sample_str[best_col]

    # 5. Apply final column order to full DataFrame
    df = df[ordered_cols]

    # 6. Row Reordering: Stable Frequency-Rank Sorting

    # Use a thread pool for parallel frequency ranking (if needed and safe)
    # But we do it sequentially per column to avoid complexity and ensure stability
    for col in ordered_cols:
        # Compute frequency rank: rank 1 = most frequent
        counts = df[col].value_counts()
        rank_map = counts.rank(ascending=False, method='min').to_dict()

        # Map values to ranks
        ranks = df[col].map(rank_map).astype(float)  # Ensure float for stability

        # Stable sort by rank (kind='stable' is default in pandas)
        df = df.iloc[ranks.argsort(kind='stable')]

        # Reset index for clean output
        df.reset_index(drop=True, inplace=True)

    return df


def _calculate_prefix_hit_score_vectorized(
    col_order: List[str],
    sample_str: pd.DataFrame,
) -> float:
    """
    Vectorized prefix hit rate calculation using string concatenation.
    Avoids loops and apply; uses pandas vectorized operations.

    Args:
        col_order: List of column names in order
        sample_str: DataFrame of string values (already converted)

    Returns:
        Average prefix hit rate (fraction of rows where current row starts with previous)
    """
    # Concatenate all columns in order
    concatenated = sample_str[col_order[0]].copy()
    for col in col_order[1:]:
        concatenated += sample_str[col]

    # Use numpy for fast string comparison
    prev = concatenated.shift(1).fillna("")
    curr = concatenated

    # Vectorized string startswith check
    # Use np.char.startswith only if we have strings; ensure dtype is object
    hits = np.char.startswith(curr.values, prev.values)  # Works on string arrays

    return hits.sum() / len(hits)