#!/usr/bin/env python3
"""Test the effect of calibration-set diversity on proxy fidelity.

Two knobs:
  1. Method (which subset selector + predictor)
  2. Calibration diversity (how spread-out are the prompts we calibrate on)

Pools 18 prompts from two IFBench init runs. For 30 random test/pool splits,
selects a "diverse" or "tight" calibration set from the pool and measures how
well each method predicts the held-out test prompts' full scores.

Run:
    python examples/ifbench/compare_calibration_diversity.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from levi.init.proxy_benchmark import select_proxy_problem_subset

RUN_A = "runs/20260419_100536_ifbench"
RUN_B = "runs/20260419_223656_ifbench"
SUBSET_K = 15
N_SPLITS = 30
TEST_SIZE = 6
CAL_SIZE = 6
RIDGE_ALPHA = 1.0
RNG_SEED = 0


def load_all_prompts() -> tuple[np.ndarray, np.ndarray]:
    matrices, fulls = [], []
    for run in (RUN_A, RUN_B):
        with open(Path(run) / "proxy_benchmark.json") as f:
            pb = json.load(f)
        rows = [p["loose_instruction_fractions"] for p in pb["prompts"]]
        full = [p["full_score"] for p in pb["prompts"]]
        matrices.append(np.asarray(rows, dtype=float))
        fulls.append(np.asarray(full, dtype=float))
    return np.vstack(matrices), np.concatenate(fulls)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> tuple[np.ndarray, float]:
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    w = np.linalg.solve(A, Xc.T @ yc)
    b = float(y.mean() - X.mean(axis=0) @ w)
    return w, b


def select_current(train_M: np.ndarray, k: int) -> list[int]:
    return select_proxy_problem_subset(train_M, k).selected_indices


def select_random(train_M: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    return rng.choice(train_M.shape[1], size=k, replace=False).tolist()


def select_kmedoids(train_M: np.ndarray, k: int, rng: np.random.Generator) -> tuple[list[int], np.ndarray]:
    points = train_M.T
    n_points = points.shape[0]
    medoid_idxs = rng.choice(n_points, size=k, replace=False).tolist()
    for _ in range(50):
        dists = np.linalg.norm(points[:, None, :] - points[medoid_idxs][None, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)
        new_medoids = []
        for c in range(k):
            cluster = np.where(assignments == c)[0]
            if len(cluster) == 0:
                new_medoids.append(medoid_idxs[c])
                continue
            sub = points[cluster]
            inner = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2).sum(axis=1)
            new_medoids.append(int(cluster[np.argmin(inner)]))
        if new_medoids == medoid_idxs:
            break
        medoid_idxs = new_medoids
    dists = np.linalg.norm(points[:, None, :] - points[medoid_idxs][None, :, :], axis=2)
    assignments = np.argmin(dists, axis=1)
    cluster_sizes = np.array([(assignments == c).sum() for c in range(k)], dtype=float)
    weights = cluster_sizes / cluster_sizes.sum()
    return medoid_idxs, weights


def pairwise_order_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    n = len(pred)
    if n < 2:
        return 1.0
    correct, total = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            d_pred = pred[i] - pred[j]
            d_true = true[i] - true[j]
            total += 1
            if abs(d_true) <= 1e-12 and abs(d_pred) <= 1e-12:
                correct += 1.0
            elif abs(d_true) <= 1e-12 or abs(d_pred) <= 1e-12:
                correct += 0.5
            elif d_pred * d_true > 0:
                correct += 1.0
    return correct / total


def spearman(pred: np.ndarray, true: np.ndarray) -> float:
    def rank(x):
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x))
        return ranks
    rp, rt = rank(pred), rank(true)
    if rp.std() == 0 or rt.std() == 0:
        return 0.0
    return float(np.corrcoef(rp, rt)[0, 1])


def top1_regret(pred: np.ndarray, true: np.ndarray) -> float:
    return float(true.max() - true[int(np.argmax(pred))])


def topk_overlap(pred: np.ndarray, true: np.ndarray, k: int = 3) -> float:
    pred_top = set(np.argsort(pred)[::-1][:k].tolist())
    true_top = set(np.argsort(true)[::-1][:k].tolist())
    return len(pred_top & true_top) / k


def evaluate(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float, float]:
    return (
        pairwise_order_accuracy(pred, true),
        spearman(pred, true),
        top1_regret(pred, true),
        topk_overlap(pred, true, k=3),
    )


def fps_select(pool_indices: list[int], distances: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """Greedy farthest-point sampling. Returns 'most diverse' subset of pool_indices."""
    pool = list(pool_indices)
    seed_idx = rng.choice(pool)
    selected = [int(seed_idx)]
    while len(selected) < k:
        remaining = [i for i in pool if i not in selected]
        # for each remaining, dist to nearest already-selected; pick max
        min_dists = []
        for r in remaining:
            min_dists.append(min(distances[r, s] for s in selected))
        best = remaining[int(np.argmax(min_dists))]
        selected.append(best)
    return selected


def tight_select(pool_indices: list[int], distances: np.ndarray, k: int) -> list[int]:
    """Select k pool prompts forming the tightest cluster (minimal sum-pairwise-distance).

    Greedy: start with the closest pair, then add the prompt that minimizes
    the sum of distances to the already-selected.
    """
    pool = list(pool_indices)
    # start with closest pair
    best_pair, best_d = None, np.inf
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            d = distances[pool[i], pool[j]]
            if d < best_d:
                best_d, best_pair = d, (pool[i], pool[j])
    selected = [int(best_pair[0]), int(best_pair[1])]
    while len(selected) < k:
        remaining = [i for i in pool if i not in selected]
        best, best_d = None, np.inf
        for r in remaining:
            d = sum(distances[r, s] for s in selected)
            if d < best_d:
                best_d, best = d, r
        selected.append(int(best))
    return selected


@dataclass
class Cell:
    method: str
    diversity: str  # "diverse" or "tight"
    pairwise: float
    spearman: float
    top1_regret: float
    top3_overlap: float
    cal_spread: float  # mean pairwise distance within calibration set (for sanity)


def run_method_on_split(
    M: np.ndarray, full: np.ndarray,
    cal_idx: list[int], test_idx: list[int],
    method_name: str,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    train_M, train_full = M[cal_idx], full[cal_idx]
    test_M, test_full = M[test_idx], full[test_idx]

    if method_name == "Current":
        sel = select_current(train_M, SUBSET_K)
        pred = test_M[:, sel].mean(axis=1)
    elif method_name == "Random + mean":
        sel = select_random(train_M, SUBSET_K, rng)
        pred = test_M[:, sel].mean(axis=1)
    elif method_name == "Random + Ridge":
        sel = select_random(train_M, SUBSET_K, rng)
        w, b = ridge_fit(train_M[:, sel], train_full)
        pred = test_M[:, sel] @ w + b
    elif method_name == "k-medoids + weighted":
        sel, weights = select_kmedoids(train_M, SUBSET_K, rng)
        pred = test_M[:, sel] @ weights
    else:
        raise ValueError(method_name)
    return evaluate(pred, test_full)


def main() -> None:
    M, full = load_all_prompts()
    n_prompts = M.shape[0]
    print(f"Pool: {n_prompts} prompts × {M.shape[1]} problems")
    print(f"Full score range: [{full.min():.4f}, {full.max():.4f}], spread {full.max()-full.min():.4f}")

    # Pairwise distances in score-vector space (across problems)
    distances = np.linalg.norm(M[:, None, :] - M[None, :, :], axis=2)
    print(f"Mean pairwise prompt distance (in 150-dim score space): {distances[np.triu_indices(n_prompts, 1)].mean():.4f}")
    print(f"  Min: {distances[np.triu_indices(n_prompts, 1)].min():.4f}, "
          f"max: {distances[np.triu_indices(n_prompts, 1)].max():.4f}")
    print()

    methods = ["Current", "Random + mean", "Random + Ridge", "k-medoids + weighted"]
    rng = np.random.default_rng(RNG_SEED)

    # Per (method, diversity) we accumulate metrics across N_SPLITS
    results: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {
        (m, d): [] for m in methods for d in ("diverse", "tight")
    }
    cal_spreads: dict[str, list[float]] = {"diverse": [], "tight": []}

    for split in range(N_SPLITS):
        all_idx = np.arange(n_prompts)
        rng.shuffle(all_idx)
        test_idx = all_idx[:TEST_SIZE].tolist()
        pool_idx = all_idx[TEST_SIZE:].tolist()

        cal_diverse = fps_select(pool_idx, distances, CAL_SIZE, rng)
        cal_tight = tight_select(pool_idx, distances, CAL_SIZE)

        spread_d = np.mean([distances[i, j] for i in cal_diverse for j in cal_diverse if i < j])
        spread_t = np.mean([distances[i, j] for i in cal_tight for j in cal_tight if i < j])
        cal_spreads["diverse"].append(float(spread_d))
        cal_spreads["tight"].append(float(spread_t))

        for method in methods:
            for diversity, cal_idx in (("diverse", cal_diverse), ("tight", cal_tight)):
                # average over a few rng draws for stochastic methods
                trial_metrics = []
                n_trials = 20 if "Random" in method or "k-medoids" in method else 1
                for t in range(n_trials):
                    method_rng = np.random.default_rng(split * 1000 + t)
                    trial_metrics.append(
                        run_method_on_split(M, full, cal_idx, test_idx, method, method_rng)
                    )
                avg = np.mean(trial_metrics, axis=0).tolist()
                results[(method, diversity)].append(tuple(avg))

    # Aggregate
    cells = []
    for (method, diversity), runs in results.items():
        arr = np.array(runs)
        means = arr.mean(axis=0)
        spread = float(np.mean(cal_spreads[diversity]))
        cells.append(Cell(method, diversity, *means.tolist(), cal_spread=spread))

    print(f"Mean calibration-set spread (intra-set pairwise distance):")
    print(f"  Diverse: {np.mean(cal_spreads['diverse']):.4f}")
    print(f"  Tight:   {np.mean(cal_spreads['tight']):.4f}")
    print()
    print_table(cells)
    save_artifacts(cells, methods, full)


def print_table(cells: list[Cell]) -> None:
    headers = ["Method", "Calibration", "Pairwise↑", "Spearman↑", "Top-1 regret↓", "Top-3 overlap↑"]
    rows = [
        [c.method, c.diversity,
         f"{c.pairwise:.3f}", f"{c.spearman:+.3f}",
         f"{c.top1_regret:.4f}", f"{c.top3_overlap:.3f}"]
        for c in cells
    ]
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("─" * w for w in widths))
    prev_method = None
    for c, row in zip(cells, rows):
        if prev_method is not None and c.method != prev_method:
            print("  ".join("·" * w for w in widths))
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        prev_method = c.method


def save_artifacts(cells: list[Cell], methods: list[str], full: np.ndarray) -> None:
    out_dir = Path("runs/proxy_method_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_splits": N_SPLITS,
        "test_size": TEST_SIZE,
        "cal_size": CAL_SIZE,
        "subset_k": SUBSET_K,
        "results": [
            {
                "method": c.method, "diversity": c.diversity,
                "pairwise": c.pairwise, "spearman": c.spearman,
                "top1_regret": c.top1_regret, "top3_overlap": c.top3_overlap,
                "cal_spread": c.cal_spread,
            }
            for c in cells
        ],
    }
    (out_dir / "calibration_diversity_results.json").write_text(json.dumps(payload, indent=2))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  Saved JSON to {out_dir/'calibration_diversity_results.json'}")
        return

    metrics = [
        ("Pairwise order accuracy (↑)", "pairwise", 0.5),
        ("Spearman ρ (↑)", "spearman", 0.0),
        ("Top-1 regret (↓)", "top1_regret", None),
        ("Top-3 overlap (↑)", "top3_overlap", 3 / TEST_SIZE),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    bar_width = 0.38
    x = np.arange(len(methods))

    for ax, (title, attr, baseline) in zip(axes.flat, metrics):
        diverse_vals = [getattr(c, attr) for c in cells if c.diversity == "diverse"]
        tight_vals = [getattr(c, attr) for c in cells if c.diversity == "tight"]
        diverse_methods = [c.method for c in cells if c.diversity == "diverse"]
        # ensure ordering matches `methods`
        d_map = dict(zip(diverse_methods, diverse_vals))
        t_methods = [c.method for c in cells if c.diversity == "tight"]
        t_map = dict(zip(t_methods, tight_vals))
        d_ordered = [d_map[m] for m in methods]
        t_ordered = [t_map[m] for m in methods]

        ax.bar(x - bar_width/2, d_ordered, bar_width, label="Diverse calibration",
               color="#2a9d8f")
        ax.bar(x + bar_width/2, t_ordered, bar_width, label="Tight calibration",
               color="#e76f51")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" + ", "\n+ ") for m in methods], fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if baseline is not None:
            ax.axhline(baseline, linestyle="--", color="gray", alpha=0.5, linewidth=1,
                       label=f"reference ({baseline:.2f})")
        for xi, (d_val, t_val) in enumerate(zip(d_ordered, t_ordered)):
            fmt = "+.3f" if "Spearman" in title else ".3f"
            ax.text(xi - bar_width/2, d_val, f"{d_val:{fmt}}", ha="center",
                    va="bottom" if d_val >= 0 else "top", fontsize=7)
            ax.text(xi + bar_width/2, t_val, f"{t_val:{fmt}}", ha="center",
                    va="bottom" if t_val >= 0 else "top", fontsize=7)
        ax.legend(loc="best", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Calibration diversity × method (K={SUBSET_K} probe problems, {N_SPLITS} random splits)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    chart_path = out_dir / "calibration_diversity.png"
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    print(f"  Saved chart to {chart_path}")
    print(f"  Saved JSON  to {out_dir/'calibration_diversity_results.json'}")


if __name__ == "__main__":
    main()
