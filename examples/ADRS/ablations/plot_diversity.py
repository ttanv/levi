#!/usr/bin/env python3
"""Visualise archive behavioral diversity across ablation conditions."""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

OUT_DIR = Path("results")

CONDITIONS = {
    "levi_full":    {"label": "LEVI (full)",      "color": "#1f77b4", "marker": "o"},
    "no_cvt":       {"label": "No Archive",        "color": "#d62728", "marker": "X"},
    "uniform_init": {"label": "Uniform Centroids", "color": "#2ca02c", "marker": "s"},
    "no_noise":     {"label": "No Init Noise",     "color": "#ff7f0e", "marker": "D"},
}

FEATURES = ["loop_nesting_max", "comparison_count", "call_count", "branch_count"]

PROBLEM_TITLES = {
    "txn": "Transaction Scheduling",
    "cbl": "Can't Be Late Scheduling",
}


def load_elites(runs_dir: Path, condition: str):
    """Load all elite behavior vectors + scores across seeds."""
    elites = []
    for d in sorted(runs_dir.glob(f"{condition}_seed*")):
        snap = json.loads((d / "snapshot.json").read_text())
        seed = int(d.name.split("seed")[1])
        for e in snap["elites"]:
            beh = [e["behavior"][f] for f in FEATURES]
            elites.append({
                "behavior": beh,
                "score": e["primary_score"],
                "seed": seed,
            })
    return elites


def make_plot(problem: str):
    runs_dir = Path(f"runs/ablations/combined/{problem}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect all behavior vectors for joint PCA ──
    all_vecs = []
    all_meta = []  # (condition, score, seed)
    for cond in CONDITIONS:
        elites = load_elites(runs_dir, cond)
        for e in elites:
            all_vecs.append(e["behavior"])
            all_meta.append((cond, e["score"], e["seed"]))

    X = np.array(all_vecs)
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 7))

    for cond, meta in CONDITIONS.items():
        mask = [i for i, m in enumerate(all_meta) if m[0] == cond]
        pts = X2[mask]

        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=meta["color"], marker=meta["marker"],
            s=60, alpha=0.7, edgecolors="white", linewidths=0.5,
            zorder=3,
            label=f"{meta['label']} ({len(mask)} programs)",
        )

    # ── Convex hulls to show spread ──
    for cond, meta in CONDITIONS.items():
        mask = [i for i, m in enumerate(all_meta) if m[0] == cond]
        pts = X2[mask]
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                        color=meta["color"], alpha=0.08, zorder=1)
                ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                        color=meta["color"], alpha=0.3, linewidth=1.5, zorder=2)
            except Exception:
                pass

    # ── Variance annotation ──
    var_text = []
    for cond, meta in CONDITIONS.items():
        mask = [i for i, m in enumerate(all_meta) if m[0] == cond]
        pts = X2[mask]
        spread = np.mean(np.std(pts, axis=0)) if len(pts) > 1 else 0
        var_text.append(f"{meta['label']}: spread={spread:.3f}")
    ax.text(0.02, 0.02, "\n".join(var_text), transform=ax.transAxes,
            fontsize=8, va="bottom", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)

    title = f"Archive Behavioral Diversity: {PROBLEM_TITLES[problem]}"
    subtitle = "PCA of 4D behavior vectors (AST features)  |  5 seeds  |  200 evals each"
    ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold")

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    out_path = OUT_DIR / f"ablation_{problem}_diversity.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    problems = sys.argv[1:] if len(sys.argv) > 1 else ["txn", "cbl"]
    for p in problems:
        make_plot(p)
