#!/usr/bin/env python3
"""Side-by-side ranked columns: ground truth vs each proxy method.

For each method, predicts full scores for all 18 original IFBench prompts using
the 6 diverse prompts as calibration. Then displays:

  - Left column: prompts ordered by true full-benchmark score (best → worst).
  - Subsequent columns: same prompts reordered under each method's prediction.

Same prompt uses the same color across all columns so you can eyeball where
it moved. A numeric side-by-side table is also printed.

Run:
    python examples/ifbench/visualize_ranking_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from levi.init.proxy_benchmark import select_proxy_problem_subset

ORIGINAL_RUNS = ["runs/20260419_100536_ifbench", "runs/20260419_223656_ifbench"]
DIVERSE_FILE = "runs/diverse_prompts/diverse_prompts.json"
SUBSET_K = 15
RIDGE_ALPHA = 1.0
SEED = 0


def load_pool():
    rows, fulls, labels = [], [], []
    for run in ORIGINAL_RUNS:
        run_id = run.split("/")[-1][:15]
        with open(Path(run) / "proxy_benchmark.json") as f:
            pb = json.load(f)
        for p in pb["prompts"]:
            rows.append(p["loose_instruction_fractions"])
            fulls.append(p["full_score"])
            labels.append(f"{run_id[:8]}:init{p['init_order']}")
    with open(DIVERSE_FILE) as f:
        diverse = json.load(f)
    for p in diverse["prompts"]:
        rows.append(p["loose_instruction_fractions"])
        fulls.append(p["full_score"])
        labels.append(f"diverse:{p['init_source']}")
    is_diverse = np.array([False] * 18 + [True] * 6)
    return np.asarray(rows), np.asarray(fulls), labels, is_diverse


def ridge_fit(X, y, alpha=RIDGE_ALPHA):
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    w = np.linalg.solve(A, Xc.T @ yc)
    b = float(y.mean() - X.mean(axis=0) @ w)
    return w, b


def select_kmedoids(M, k, rng):
    points = M.T
    n = points.shape[0]
    medoids = rng.choice(n, size=k, replace=False).tolist()
    for _ in range(50):
        d = np.linalg.norm(points[:, None] - points[medoids][None, :], axis=2)
        a = np.argmin(d, axis=1)
        new = []
        for c in range(k):
            cluster = np.where(a == c)[0]
            if len(cluster) == 0:
                new.append(medoids[c])
                continue
            sub = points[cluster]
            inner = np.linalg.norm(sub[:, None] - sub[None, :], axis=2).sum(axis=1)
            new.append(int(cluster[np.argmin(inner)]))
        if new == medoids:
            break
        medoids = new
    d = np.linalg.norm(points[:, None] - points[medoids][None, :], axis=2)
    a = np.argmin(d, axis=1)
    sizes = np.array([(a == c).sum() for c in range(k)], dtype=float)
    return medoids, sizes / sizes.sum()


def predict_for_pool(method, M, full, cal_idx, test_idx, rng):
    train_M, train_full = M[cal_idx], full[cal_idx]
    test_M = M[test_idx]
    if method == "Current":
        sel = select_proxy_problem_subset(train_M, SUBSET_K).selected_indices
        return test_M[:, sel].mean(axis=1)
    if method == "Random + mean":
        sel = rng.choice(train_M.shape[1], size=SUBSET_K, replace=False).tolist()
        return test_M[:, sel].mean(axis=1)
    if method == "Random + Ridge":
        sel = rng.choice(train_M.shape[1], size=SUBSET_K, replace=False).tolist()
        w, b = ridge_fit(train_M[:, sel], train_full)
        return test_M[:, sel] @ w + b
    if method == "k-medoids":
        sel, weights = select_kmedoids(train_M, SUBSET_K, rng)
        return test_M[:, sel] @ weights
    raise ValueError(method)


def main():
    M, full, labels, is_diverse = load_pool()
    orig_idx = np.where(~is_diverse)[0]
    div_idx = np.where(is_diverse)[0]

    methods = ["Current", "Random + mean", "Random + Ridge", "k-medoids"]

    # Predict full scores for all 18 originals using 6 diverse as calibration
    preds = {}
    for m in methods:
        rng = np.random.default_rng(SEED)
        preds[m] = predict_for_pool(m, M, full, div_idx.tolist(), orig_idx.tolist(), rng)

    true_scores = full[orig_idx]
    true_labels = [labels[i].split(":", 1)[-1] for i in orig_idx]  # just "initN"
    true_full_labels = [labels[i] for i in orig_idx]

    print_table(true_labels, true_scores, preds, methods)
    save_plot(true_labels, true_scores, preds, methods)


def rank_desc(scores):
    """Return 1-based rank per element (1 = highest score)."""
    return np.argsort(np.argsort(-scores)) + 1


def print_table(prompt_labels, true_scores, preds, methods):
    true_rank = rank_desc(true_scores)
    order = np.argsort(-true_scores)

    print("\nSide-by-side ranking of all 18 originals (sorted by true full score)")
    print("=" * 100)
    header = f"{'Prompt':24s} | {'True':>6s} |"
    for m in methods:
        header += f" {m[:13]:>13s} |"
    print(header)
    print("-" * len(header))
    for i in order:
        row = f"{prompt_labels[i]:24s} | {true_scores[i]:.4f} |"
        for m in methods:
            pr = int(rank_desc(preds[m])[i])
            shift = pr - int(true_rank[i])
            marker = f"+{shift}" if shift > 0 else str(shift)
            row += f"  #{pr:>2d}/18 ({marker:>3s}) |"
        print(row)

    # Top-5 overlap summary
    print()
    print("Top-5 overlap with ground truth:")
    gt_top5 = set(order[:5].tolist())
    for m in methods:
        m_top5 = set(np.argsort(-preds[m])[:5].tolist())
        overlap = len(gt_top5 & m_top5)
        print(f"  {m:20s}  {overlap}/5")


def save_plot(prompt_labels, true_scores, preds, methods):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        return

    n = len(prompt_labels)
    columns = ["True"] + methods

    # Consistent color per prompt, based on true rank
    true_rank = rank_desc(true_scores)
    cmap = plt.get_cmap("viridis")
    # prompt i's color = cmap((true_rank[i]-1)/(n-1))
    prompt_colors = {i: cmap((true_rank[i] - 1) / (n - 1)) for i in range(n)}

    # Build per-column ordering: for each column, list of prompt indices in rank order
    col_orders = []
    for col in columns:
        if col == "True":
            col_orders.append(np.argsort(-true_scores).tolist())
        else:
            col_orders.append(np.argsort(-preds[col]).tolist())

    col_w = 2.6  # horizontal spacing between columns
    row_h = 1.0
    box_w = 2.2
    box_h = 0.82

    fig_w = max(12, col_w * len(columns) + 3)
    fig_h = row_h * n + 2.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for col_i, (col_name, ordering) in enumerate(zip(columns, col_orders)):
        for row_i, prompt_i in enumerate(ordering):
            color = prompt_colors[prompt_i]
            x = col_i * col_w
            y = (n - row_i) * row_h
            box = FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=0.8,
                edgecolor="white",
                facecolor=color,
                alpha=0.9,
            )
            ax.add_patch(box)
            text_color = "white" if true_rank[prompt_i] <= n / 2 else "black"
            label_str = prompt_labels[prompt_i]
            if col_name == "True":
                label_str = f"{label_str}  ({true_scores[prompt_i]:.3f})"
            else:
                # show rank-shift so you see drift at a glance
                pr = int(rank_desc(preds[col_name])[prompt_i])
                tr = int(true_rank[prompt_i])
                shift = pr - tr
                if shift == 0:
                    shift_str = ""
                elif shift > 0:
                    shift_str = f"  (↓{shift})"
                else:
                    shift_str = f"  (↑{-shift})"
                label_str = f"{label_str}{shift_str}"
            ax.text(x, y, label_str, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    # Rank labels on left
    for row_i in range(n):
        ax.text(-col_w * 0.55, (n - row_i) * row_h, f"#{row_i + 1}",
                ha="right", va="center", fontsize=11,
                fontweight="bold", color="#444")

    # Column headers
    header_y = (n + 1) * row_h
    for col_i, col_name in enumerate(columns):
        ax.text(col_i * col_w, header_y, col_name,
                ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_xlim(-col_w * 0.9, (len(columns) - 1) * col_w + col_w / 2)
    ax.set_ylim(0.2, header_y + 1.0)
    ax.axis("off")
    fig.suptitle(
        "Ground-truth ranking vs each method's ranking  (18 originals, calibration = 6 diverse prompts)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout()

    out_dir = Path("runs/proxy_method_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ranking_columns.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
