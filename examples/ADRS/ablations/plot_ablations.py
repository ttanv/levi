#!/usr/bin/env python3
"""Generate academic ablation plots matching comparison_plot.png style."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RUNS_DIR = Path("runs/ablations/combined")
OUT_DIR = Path("results")

CONDITIONS = {
    "levi_full":    {"label": "LEVI (full)",      "color": "#1f77b4"},
    "no_cvt":       {"label": "No Archive",        "color": "#d62728"},
    "uniform_init": {"label": "Uniform Centroids", "color": "#2ca02c"},
    "no_noise":     {"label": "No Init Noise",     "color": "#ff7f0e"},
}

PROBLEMS = {
    "txn": {
        "title": "Transaction Scheduling",
        "seed_score": 41.9,
    },
    "cbl": {
        "title": "Can't Be Late Scheduling",
        "seed_score": 79.5,
    },
}


def load_best_so_far_curves(problem: str, condition: str):
    """Load best-so-far curves for all seeds of a (problem, condition) pair."""
    base = RUNS_DIR / problem
    curves = []
    for d in sorted(base.glob(f"{condition}_seed*")):
        snap = d / "snapshot.json"
        if not snap.exists():
            continue
        data = json.loads(snap.read_text())
        history = data.get("score_history", [])
        if not history:
            continue
        eval_nums = [h["eval_number"] for h in history]
        best_scores = [h["best_score"] for h in history]
        curves.append((eval_nums, best_scores))
    return curves


def interpolate_curves(curves, max_eval=200):
    """Interpolate all curves onto a common eval grid and return mean/std."""
    x_grid = np.arange(1, max_eval + 1)
    interp = []
    for eval_nums, best_scores in curves:
        y = np.interp(x_grid, eval_nums, best_scores)
        interp.append(y)
    interp = np.array(interp)
    return x_grid, np.mean(interp, axis=0), np.std(interp, axis=0), interp


def make_plot(problem_key: str):
    prob = PROBLEMS[problem_key]
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Collect final scores for legend
    legend_info = []

    for cond_key, cond_meta in CONDITIONS.items():
        curves = load_best_so_far_curves(problem_key, cond_key)
        if not curves:
            continue

        x, mean, std, raw = interpolate_curves(curves)
        n = len(curves)
        best_mean = mean[-1]

        ax.plot(x, mean, color=cond_meta["color"], linewidth=2.2, zorder=3)
        ax.fill_between(x, mean - std, mean + std,
                        color=cond_meta["color"], alpha=0.15, zorder=2)

        legend_info.append((cond_meta["label"], n, best_mean, cond_meta["color"]))

    # Seed baseline
    ax.axhline(prob["seed_score"], color="gray", linestyle="--", linewidth=1, alpha=0.6, zorder=1)
    ax.text(5, prob["seed_score"] + 0.5, "seed baseline", fontsize=8,
            color="gray", va="bottom")

    # Legend
    from matplotlib.lines import Line2D
    handles = []
    for label, n, best, color in legend_info:
        handles.append(Line2D([0], [0], color=color, linewidth=2.2,
                              label=f"{label} (n={n}, best: {best:.1f})"))
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    # Labels & formatting
    ax.set_xlabel("Successful Evaluations", fontsize=11)
    ax.set_ylabel("Best Score (0-100)", fontsize=11)

    title = f"Ablation Study: {prob['title']}"
    subtitle = "Qwen 30B  |  200 evaluations  |  5 seeds  |  mean \u00b1 std"
    ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold")

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    out_path = OUT_DIR / f"ablation_{problem_key}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in PROBLEMS:
        make_plot(p)
    print("Done.")
