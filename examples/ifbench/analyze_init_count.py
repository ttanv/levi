#!/usr/bin/env python3
"""How does the number of init prompts affect proxy-subset quality?

Uses the saved 9-prompt × 150-problem matrix from a completed run.
For each N in {2..8}: draw K random N-prompt subsets, run the proxy
selector, then measure ranking agreement on the held-out prompts.

Reports mean and std of out-of-sample ranking agreement to show how
many init prompts you actually need.
"""

from __future__ import annotations

import json
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# Make levi importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from levi.init.proxy_benchmark import (  # noqa: E402
    _pairwise_order_accuracy,
    select_proxy_problem_subset,
)

RUN_DIR = Path("runs/20260419_100536_ifbench")
SUBSET_SIZE = 15
SEED = 0


def main() -> None:
    pb = json.loads((RUN_DIR / "proxy_benchmark.json").read_text())
    matrix = np.array(
        [p["loose_instruction_fractions"] for p in pb["prompts"]],
        dtype=float,
    )
    n_prompts, n_problems = matrix.shape
    full_prompt_means = matrix.mean(axis=1)
    print(f"Matrix: {n_prompts} prompts × {n_problems} problems")
    print(f"Full prompt scores: {[round(x, 4) for x in full_prompt_means.tolist()]}")
    print()

    rng = random.Random(SEED)
    print(f"{'N':>3} {'#subs':>6} {'in_rank':>10} {'oos_rank':>10} {'oos_std':>10} {'mean_overlap':>14}")
    print("-" * 60)

    for n in range(2, n_prompts):
        all_combos = list(combinations(range(n_prompts), n))
        # Cap at 25 random combos for speed; otherwise enumerate all if smaller
        if len(all_combos) > 25:
            sampled = rng.sample(all_combos, 25)
        else:
            sampled = all_combos

        in_ranks: list[float] = []
        oos_ranks: list[float] = []
        selections: list[set[int]] = []

        for combo in sampled:
            subset_idx = list(combo)
            holdout_idx = [i for i in range(n_prompts) if i not in subset_idx]
            sel = select_proxy_problem_subset(matrix[subset_idx], SUBSET_SIZE)
            selected_problems = sel.selected_indices
            selections.append(set(selected_problems))

            # In-sample ranking agreement (on selection prompts)
            in_ranks.append(sel.final_ranking_score)

            # Out-of-sample ranking agreement (on held-out prompts)
            if len(holdout_idx) >= 2:
                holdout_full = matrix[holdout_idx].mean(axis=1)
                holdout_proxy = matrix[holdout_idx][:, selected_problems].mean(axis=1)
                oos_ranks.append(_pairwise_order_accuracy(holdout_full, holdout_proxy))

        # Selection stability: mean Jaccard overlap between selected sets
        overlaps = []
        for i in range(len(selections)):
            for j in range(i + 1, len(selections)):
                a, b = selections[i], selections[j]
                overlaps.append(len(a & b) / len(a | b))
        mean_overlap = float(np.mean(overlaps)) if overlaps else 1.0

        in_mean = float(np.mean(in_ranks))
        oos_mean = float(np.mean(oos_ranks)) if oos_ranks else float("nan")
        oos_std = float(np.std(oos_ranks)) if oos_ranks else float("nan")

        print(
            f"{n:>3} {len(sampled):>6} {in_mean:>10.4f} {oos_mean:>10.4f} {oos_std:>10.4f} {mean_overlap:>14.3f}"
        )

    print()
    print("Legend:")
    print("  in_rank  = pairwise rank-agreement on prompts used for selection (always near 1.0)")
    print("  oos_rank = pairwise rank-agreement on held-out prompts (the real generalization metric)")
    print("  oos_std  = std-dev of oos_rank across random subsets — high = unreliable")
    print("  mean_overlap = Jaccard overlap of the 15 chosen problems across subsets — high = stable picks")


if __name__ == "__main__":
    main()
