#!/usr/bin/env python3
"""Generate a paper-quality plot of how init-prompt count affects proxy quality."""

from __future__ import annotations

import json
import random
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from levi.init.proxy_benchmark import (  # noqa: E402
    _pairwise_order_accuracy,
    select_proxy_problem_subset,
)

RUN_DIR = Path("runs/20260419_100536_ifbench")
SUBSET_SIZE = 15
SEED = 0
MAX_COMBOS = 25
OUT_PATH = Path("examples/ifbench/proxy_init_count.pdf")


def gather() -> dict[int, dict[str, float]]:
    """For each N, sample N-prompt subsets, build proxy, measure quality."""
    pb = json.loads((RUN_DIR / "proxy_benchmark.json").read_text())
    matrix = np.array(
        [p["loose_instruction_fractions"] for p in pb["prompts"]],
        dtype=float,
    )
    n_prompts = matrix.shape[0]
    full_prompt_scores = matrix.mean(axis=1)
    rng = random.Random(SEED)
    out: dict[int, dict[str, float]] = {}

    for n in range(2, n_prompts + 1):
        all_combos = list(combinations(range(n_prompts), n))
        sampled = rng.sample(all_combos, min(MAX_COMBOS, len(all_combos)))

        rank_agreements: list[float] = []
        selections: list[set[int]] = []

        for combo in sampled:
            subset_idx = list(combo)
            sel = select_proxy_problem_subset(matrix[subset_idx], SUBSET_SIZE)
            selections.append(set(sel.selected_indices))
            # Rank agreement on ALL 9 prompts: how well does the proxy
            # (selected from N prompts) reproduce the full-benchmark ranking?
            proxy_scores = matrix[:, sel.selected_indices].mean(axis=1)
            rank_agreements.append(
                _pairwise_order_accuracy(full_prompt_scores, proxy_scores)
            )

        overlaps: list[float] = []
        for i in range(len(selections)):
            for j in range(i + 1, len(selections)):
                a, b = selections[i], selections[j]
                overlaps.append(len(a & b) / len(a | b))

        out[n] = {
            "rank_mean": float(np.mean(rank_agreements)),
            "rank_std": float(np.std(rank_agreements)),
            "stability": float(np.mean(overlaps)) if overlaps else 1.0,
        }
    return out


def main() -> None:
    data = gather()
    ns = sorted(data.keys())

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    rank_means = np.array([data[n]["rank_mean"] for n in ns])
    rank_stds = np.array([data[n]["rank_std"] for n in ns])
    stab = np.array([data[n]["stability"] for n in ns])

    color_rank = "#1f77b4"
    color_stab = "#888888"

    # Rank agreement (primary metric) with shaded std band
    ax.fill_between(
        ns, rank_means - rank_stds, rank_means + rank_stds,
        color=color_rank, alpha=0.15, linewidth=0,
    )
    ax.plot(
        ns, rank_means,
        marker="o", color=color_rank, linewidth=2.0, markersize=6,
        label="Rank agreement vs. full benchmark",
    )

    # Selection stability (secondary, same axis since both are in [0,1])
    ax.plot(
        ns, stab,
        marker="s", linestyle="--", color=color_stab, linewidth=1.4, markersize=5,
        label="Selection stability (Jaccard)",
    )

    ax.set_xlabel("Init prompts used to build proxy benchmark (N)")
    ax.set_ylabel("Score")
    ax.set_xticks(ns)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"Wrote {OUT_PATH} and {OUT_PATH.with_suffix('.png')}")

    print("\nValues:")
    print(f"{'N':>3} {'rank':>8} {'±std':>8} {'jaccard':>9}")
    for n in ns:
        d = data[n]
        print(f"{n:>3} {d['rank_mean']:>8.3f} {d['rank_std']:>8.3f} {d['stability']:>9.3f}")


if __name__ == "__main__":
    main()
