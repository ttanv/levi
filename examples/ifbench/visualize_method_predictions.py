#!/usr/bin/env python3
"""Visualize predicted vs actual scores for each proxy method.

Pools 18 original + 6 diverse IFBench prompts (24 total). For each random
split (test = 6 held-out originals, cal = 6 diverse prompts), runs each method
and collects (predicted, actual) pairs. Produces:
  1. Scatter plots: predicted vs actual per method (aggregated over splits).
  2. A concrete single-split example table showing how each method ranks test prompts.

Run:
    python examples/ifbench/visualize_method_predictions.py
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
N_SPLITS = 80
RIDGE_ALPHA = 1.0


def load_pool():
    rows, fulls, labels = [], [], []
    for run in ORIGINAL_RUNS:
        run_id = run.split("/")[-1][:15]
        with open(Path(run) / "proxy_benchmark.json") as f:
            pb = json.load(f)
        for p in pb["prompts"]:
            rows.append(p["loose_instruction_fractions"])
            fulls.append(p["full_score"])
            labels.append(f"{run_id}:init{p['init_order']}")

    with open(DIVERSE_FILE) as f:
        diverse = json.load(f)
    for p in diverse["prompts"]:
        rows.append(p["loose_instruction_fractions"])
        fulls.append(p["full_score"])
        labels.append(f"diverse:{p['init_source']}")

    return (
        np.asarray(rows, dtype=float),
        np.asarray(fulls, dtype=float),
        labels,
        # is_diverse mask: True for the last 6
        np.array([False] * 18 + [True] * 6),
    )


def ridge_fit(X, y, alpha=RIDGE_ALPHA):
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    w = np.linalg.solve(A, Xc.T @ yc)
    b = float(y.mean() - X.mean(axis=0) @ w)
    return w, b


def select_current(M, k):
    return select_proxy_problem_subset(M, k).selected_indices


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


def run_method(method, M, full, cal_idx, test_idx, rng):
    train_M, train_full = M[cal_idx], full[cal_idx]
    test_M = M[test_idx]
    if method == "Current":
        sel = select_current(train_M, SUBSET_K)
        pred = test_M[:, sel].mean(axis=1)
    elif method == "Random + mean":
        sel = rng.choice(train_M.shape[1], size=SUBSET_K, replace=False).tolist()
        pred = test_M[:, sel].mean(axis=1)
    elif method == "Random + Ridge":
        sel = rng.choice(train_M.shape[1], size=SUBSET_K, replace=False).tolist()
        w, b = ridge_fit(train_M[:, sel], train_full)
        pred = test_M[:, sel] @ w + b
    elif method == "k-medoids + weighted":
        sel, weights = select_kmedoids(train_M, SUBSET_K, rng)
        pred = test_M[:, sel] @ weights
    else:
        raise ValueError(method)
    return pred


def main():
    M, full, labels, is_diverse = load_pool()
    orig_idx = np.where(~is_diverse)[0]
    div_idx = np.where(is_diverse)[0]

    methods = ["Current", "Random + mean", "Random + Ridge", "k-medoids + weighted"]

    # Aggregate: for each method, collect all (predicted, actual) pairs across splits
    agg: dict[str, list[tuple[float, float]]] = {m: [] for m in methods}
    rng = np.random.default_rng(0)
    for split in range(N_SPLITS):
        perm = rng.permutation(orig_idx)
        test_idx = perm[:6].tolist()
        cal_idx = div_idx.tolist()
        for method in methods:
            n_trials = 20 if method != "Current" else 1
            for t in range(n_trials):
                method_rng = np.random.default_rng(split * 1000 + t)
                pred = run_method(method, M, full, cal_idx, test_idx, method_rng)
                actual = full[test_idx]
                for p, a in zip(pred.tolist(), actual.tolist()):
                    agg[method].append((p, a))

    # Concrete single-split example
    example_rng = np.random.default_rng(42)
    perm = example_rng.permutation(orig_idx)
    ex_test_idx = perm[:6].tolist()
    ex_cal_idx = div_idx.tolist()
    example_preds = {}
    for method in methods:
        m_rng = np.random.default_rng(42)
        example_preds[method] = run_method(method, M, full, ex_cal_idx, ex_test_idx, m_rng)

    print_example_table(ex_test_idx, labels, full, example_preds, methods)
    save_visualizations(agg, methods, ex_test_idx, labels, full, example_preds)


def print_example_table(test_idx, labels, full, preds, methods):
    print("=" * 110)
    print("CONCRETE EXAMPLE: one random split, test = 6 held-out originals, cal = 6 diverse prompts")
    print("=" * 110)
    print()

    # Actual ranking (1 = best)
    actual_scores = full[test_idx]
    actual_rank = np.argsort(np.argsort(-actual_scores)) + 1

    # Header
    header = f"{'Prompt':45s} | {'True':>6s} | {'Rank':>4s}"
    for m in methods:
        header += f" | {m[:18]:>18s}"
    print(header)
    print("-" * len(header))

    for i, tidx in enumerate(test_idx):
        short = labels[tidx][:43]
        row = f"{short:45s} | {actual_scores[i]:.4f} | {actual_rank[i]:>4d}"
        for m in methods:
            p = preds[m][i]
            # predicted rank
            pred_rank = (np.argsort(np.argsort(-preds[m])) + 1)[i]
            rank_shift = pred_rank - actual_rank[i]
            shift_str = f"+{rank_shift}" if rank_shift > 0 else str(rank_shift)
            row += f" | {p:.4f} r{pred_rank}({shift_str:>3s})"
        print(row)

    print()
    print("Reading: 'predicted_score rRANK(shift_from_true)'. r3(+1) means method ranked it 3rd; true was 2nd.")
    print()

    # Best-pick summary
    true_best_i = int(np.argmax(actual_scores))
    true_best_label = labels[test_idx[true_best_i]]
    true_best_score = actual_scores[true_best_i]
    print(f"True best: {true_best_label} (score {true_best_score:.4f})")
    print("Each method's pick as #1:")
    for m in methods:
        picked_i = int(np.argmax(preds[m]))
        picked_label = labels[test_idx[picked_i]]
        picked_actual = actual_scores[picked_i]
        regret = true_best_score - picked_actual
        flag = "✓" if picked_i == true_best_i else f"✗ (regret {regret:.4f})"
        print(f"  {m:22s}  → {picked_label[:45]:45s} actual={picked_actual:.4f}  {flag}")
    print()


def save_visualizations(agg, methods, ex_test_idx, labels, full, example_preds):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir = Path("runs/proxy_method_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Aggregate predicted vs actual scatter ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, method in zip(axes.flat, methods):
        pairs = np.array(agg[method])
        preds, actuals = pairs[:, 0], pairs[:, 1]
        # Normalize predictions to share axes with actuals — useful for Ridge that outputs
        # on the full-score scale anyway, but random+mean etc are on the proxy scale.
        # Standardize to z-scores for visual comparability:
        pred_norm = (preds - preds.mean()) / max(preds.std(), 1e-9)
        actual_norm = (actuals - actuals.mean()) / max(actuals.std(), 1e-9)
        # hexbin for density
        ax.hexbin(actual_norm, pred_norm, gridsize=25, cmap="Blues", mincnt=1, alpha=0.6)
        lim = [min(actual_norm.min(), pred_norm.min()) - 0.2,
               max(actual_norm.max(), pred_norm.max()) + 0.2]
        ax.plot(lim, lim, "k--", linewidth=1, alpha=0.4, label="perfect prediction")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Actual full score (z-scored)")
        ax.set_ylabel("Predicted proxy score (z-scored)")

        # Correlations
        pearson = float(np.corrcoef(preds, actuals)[0, 1])
        ax.set_title(f"{method}\nPearson r = {pearson:+.3f}  (n={len(pairs)} preds across splits)",
                     fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Predicted vs actual score (pooled across 80 random splits, diverse calibration)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    scatter_path = out_dir / "predictions_scatter.png"
    fig.savefig(scatter_path, dpi=120, bbox_inches="tight")
    print(f"Saved scatter plot to {scatter_path}")

    # --- Plot 2: Concrete single-split ranking comparison ---
    fig, ax = plt.subplots(figsize=(14, 7))
    actual_scores = full[ex_test_idx]
    actual_rank = np.argsort(np.argsort(-actual_scores)) + 1
    n = len(ex_test_idx)

    x_positions = {}
    col_labels = ["True"] + methods
    x_positions_list = list(range(len(col_labels)))

    # For each column (ranking by method), place prompts in rank order top->bottom
    for col, label in enumerate(col_labels):
        if label == "True":
            ranks = actual_rank
        else:
            preds = example_preds[label]
            ranks = np.argsort(np.argsort(-preds)) + 1
        for i in range(n):
            x_positions.setdefault(i, []).append((col, int(ranks[i])))

    # color by true rank
    cmap = plt.get_cmap("viridis_r")
    colors = [cmap(r / n) for r in actual_rank]

    for i in range(n):
        xs = [p[0] for p in x_positions[i]]
        ys = [p[1] for p in x_positions[i]]
        ax.plot(xs, ys, "-o", color=colors[i], alpha=0.7, markersize=10, linewidth=2)
        # label the true column with short name
        short = labels[ex_test_idx[i]].split(":")[-1][:22]
        ax.annotate(short, (xs[0] - 0.1, ys[0]), ha="right", va="center", fontsize=9, color=colors[i])

    ax.set_xticks(x_positions_list)
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels([f"#{i}" for i in range(1, n + 1)])
    ax.set_ylim(n + 0.5, 0.5)  # inverted so #1 is top
    ax.set_ylabel("Rank", fontsize=11)
    ax.set_title("How each method ranks 6 test prompts (one split; lines show where each prompt moves)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for col in x_positions_list[:-1]:
        ax.axvline(col + 0.5, color="gray", alpha=0.2, linestyle=":")

    fig.tight_layout()
    bump_path = out_dir / "ranking_bump.png"
    fig.savefig(bump_path, dpi=120, bbox_inches="tight")
    print(f"Saved ranking bump chart to {bump_path}")


if __name__ == "__main__":
    main()
