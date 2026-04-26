#!/usr/bin/env python3
"""Diversity ablation: tight vs diverse calibration on a fixed test set.

Pool: 18 original (tight) prompts + 6 deliberately-diverse prompts = 24 total.
For each random split:
  - Test = 6 random originals
  - Low-diversity cal = 6 OTHER random originals
  - High-diversity cal = the 6 fixed diverse prompts
For each method × diversity condition: predict test-prompt full scores, measure.

Run:
    python examples/ifbench/compare_with_diverse.py
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

ORIGINAL_RUNS = ["runs/20260419_100536_ifbench", "runs/20260419_223656_ifbench"]
DIVERSE_FILE = "runs/diverse_prompts/diverse_prompts.json"
SUBSET_K = 15
N_SPLITS = 50
TEST_SIZE = 6
CAL_SIZE = 6
RIDGE_ALPHA = 1.0
N_TRIALS_STOCHASTIC = 20


def load_pool() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (matrix, full_scores, is_diverse_mask)."""
    rows, fulls, is_diverse = [], [], []
    for run in ORIGINAL_RUNS:
        with open(Path(run) / "proxy_benchmark.json") as f:
            pb = json.load(f)
        for p in pb["prompts"]:
            rows.append(p["loose_instruction_fractions"])
            fulls.append(p["full_score"])
            is_diverse.append(False)

    diverse_path = Path(DIVERSE_FILE)
    if not diverse_path.exists():
        raise SystemExit(
            f"Diverse prompts not found at {diverse_path}. "
            f"Run examples/ifbench/score_diverse_prompts.py first."
        )
    with open(diverse_path) as f:
        diverse = json.load(f)
    for p in diverse["prompts"]:
        rows.append(p["loose_instruction_fractions"])
        fulls.append(p["full_score"])
        is_diverse.append(True)

    return (
        np.asarray(rows, dtype=float),
        np.asarray(fulls, dtype=float),
        np.asarray(is_diverse, dtype=bool),
    )


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA):
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    w = np.linalg.solve(A, Xc.T @ yc)
    b = float(y.mean() - X.mean(axis=0) @ w)
    return w, b


def select_current(M: np.ndarray, k: int) -> list[int]:
    return select_proxy_problem_subset(M, k).selected_indices


def select_random(M: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    return rng.choice(M.shape[1], size=k, replace=False).tolist()


def select_kmedoids(M: np.ndarray, k: int, rng: np.random.Generator):
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


def pairwise_order_accuracy(pred, true) -> float:
    n = len(pred)
    if n < 2:
        return 1.0
    correct, total = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            d_pred, d_true = pred[i] - pred[j], true[i] - true[j]
            total += 1
            if abs(d_true) <= 1e-12 and abs(d_pred) <= 1e-12:
                correct += 1.0
            elif abs(d_true) <= 1e-12 or abs(d_pred) <= 1e-12:
                correct += 0.5
            elif d_pred * d_true > 0:
                correct += 1.0
    return correct / total


def spearman(pred, true) -> float:
    def rank(x):
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x))
        return ranks
    rp, rt = rank(pred), rank(true)
    if rp.std() == 0 or rt.std() == 0:
        return 0.0
    return float(np.corrcoef(rp, rt)[0, 1])


def top1_regret(pred, true) -> float:
    return float(true.max() - true[int(np.argmax(pred))])


def topk_overlap(pred, true, k: int = 3) -> float:
    return len(set(np.argsort(pred)[::-1][:k].tolist()) & set(np.argsort(true)[::-1][:k].tolist())) / k


def evaluate(pred, true) -> tuple[float, float, float, float]:
    return (
        pairwise_order_accuracy(pred, true),
        spearman(pred, true),
        top1_regret(pred, true),
        topk_overlap(pred, true, k=3),
    )


def run_method(M, full, cal_idx, test_idx, method, rng) -> tuple[float, float, float, float]:
    train_M, train_full = M[cal_idx], full[cal_idx]
    test_M, test_full = M[test_idx], full[test_idx]
    if method == "Current":
        sel = select_current(train_M, SUBSET_K)
        pred = test_M[:, sel].mean(axis=1)
    elif method == "Random + mean":
        sel = select_random(train_M, SUBSET_K, rng)
        pred = test_M[:, sel].mean(axis=1)
    elif method == "Random + Ridge":
        sel = select_random(train_M, SUBSET_K, rng)
        w, b = ridge_fit(train_M[:, sel], train_full)
        pred = test_M[:, sel] @ w + b
    elif method == "k-medoids + weighted":
        sel, weights = select_kmedoids(train_M, SUBSET_K, rng)
        pred = test_M[:, sel] @ weights
    else:
        raise ValueError(method)
    return evaluate(pred, test_full)


@dataclass
class Cell:
    method: str
    diversity: str
    pairwise: float
    spearman: float
    top1_regret: float
    top3_overlap: float


def main() -> None:
    M, full, is_diverse = load_pool()
    print(f"Pool: {M.shape[0]} prompts ({(~is_diverse).sum()} original + {is_diverse.sum()} diverse)")
    print(f"Original score range: [{full[~is_diverse].min():.4f}, {full[~is_diverse].max():.4f}]")
    print(f"Diverse score range:  [{full[is_diverse].min():.4f}, {full[is_diverse].max():.4f}]")
    print(f"Combined range:       [{full.min():.4f}, {full.max():.4f}]")

    distances = np.linalg.norm(M[:, None] - M[None, :], axis=2)
    orig_idx = np.where(~is_diverse)[0]
    div_idx = np.where(is_diverse)[0]
    iu = np.triu_indices(len(orig_idx), 1)
    orig_pairs = distances[np.ix_(orig_idx, orig_idx)][iu]
    div_iu = np.triu_indices(len(div_idx), 1)
    div_pairs = distances[np.ix_(div_idx, div_idx)][div_iu]
    print(f"Mean intra-pool distance — original: {orig_pairs.mean():.4f}, diverse: {div_pairs.mean():.4f}")
    print()

    methods = ["Current", "Random + mean", "Random + Ridge", "k-medoids + weighted"]
    rng = np.random.default_rng(0)

    results: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {
        (m, d): [] for m in methods for d in ("low", "high")
    }

    for split in range(N_SPLITS):
        # test = 6 random originals; low-cal = 6 other random originals
        perm = rng.permutation(orig_idx)
        test_idx = perm[:TEST_SIZE].tolist()
        low_cal_idx = perm[TEST_SIZE:TEST_SIZE + CAL_SIZE].tolist()
        # high-cal = all 6 diverse prompts
        high_cal_idx = div_idx.tolist()

        for method in methods:
            for diversity, cal_idx in (("low", low_cal_idx), ("high", high_cal_idx)):
                trials = []
                n_trials = N_TRIALS_STOCHASTIC if method != "Current" else 1
                for t in range(n_trials):
                    method_rng = np.random.default_rng(split * 1000 + t)
                    trials.append(run_method(M, full, cal_idx, test_idx, method, method_rng))
                results[(method, diversity)].append(np.mean(trials, axis=0).tolist())

    cells = []
    for (method, diversity), runs in results.items():
        means = np.array(runs).mean(axis=0).tolist()
        cells.append(Cell(method, diversity, *means))

    print_table(cells)
    save_artifacts(cells, methods)


def print_table(cells: list[Cell]) -> None:
    headers = ["Method", "Calibration", "Pairwise↑", "Spearman↑", "Top-1 regret↓", "Top-3 overlap↑"]
    rows = [
        [c.method, "low-div" if c.diversity == "low" else "HIGH-div",
         f"{c.pairwise:.3f}", f"{c.spearman:+.3f}",
         f"{c.top1_regret:.4f}", f"{c.top3_overlap:.3f}"]
        for c in cells
    ]
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("─" * w for w in widths))
    prev = None
    for c, row in zip(cells, rows):
        if prev is not None and c.method != prev:
            print("  ".join("·" * w for w in widths))
        print("  ".join(s.ljust(widths[i]) for i, s in enumerate(row)))
        prev = c.method
    print()
    print("Δ (high-div − low-div):")
    for method in {c.method for c in cells}:
        low = next(c for c in cells if c.method == method and c.diversity == "low")
        high = next(c for c in cells if c.method == method and c.diversity == "high")
        print(
            f"  {method:24s}  pairwise {high.pairwise - low.pairwise:+.3f}  "
            f"spearman {high.spearman - low.spearman:+.3f}  "
            f"regret {high.top1_regret - low.top1_regret:+.4f}  "
            f"overlap {high.top3_overlap - low.top3_overlap:+.3f}"
        )


def save_artifacts(cells: list[Cell], methods: list[str]) -> None:
    out_dir = Path("runs/proxy_method_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "method": c.method, "diversity": c.diversity,
            "pairwise": c.pairwise, "spearman": c.spearman,
            "top1_regret": c.top1_regret, "top3_overlap": c.top3_overlap,
        }
        for c in cells
    ]
    (out_dir / "with_diverse_results.json").write_text(json.dumps(payload, indent=2))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    metrics = [
        ("Pairwise order accuracy (↑)", "pairwise", 0.5),
        ("Spearman ρ (↑)", "spearman", 0.0),
        ("Top-1 regret (↓)", "top1_regret", None),
        ("Top-3 overlap (↑)", "top3_overlap", 3 / TEST_SIZE),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    bar_w = 0.38
    x = np.arange(len(methods))
    for ax, (title, attr, baseline) in zip(axes.flat, metrics):
        low_vals = [getattr(c, attr) for m in methods for c in cells if c.method == m and c.diversity == "low"]
        high_vals = [getattr(c, attr) for m in methods for c in cells if c.method == m and c.diversity == "high"]
        ax.bar(x - bar_w / 2, low_vals, bar_w, label="Low-diversity calibration (6 tight prompts)", color="#e76f51")
        ax.bar(x + bar_w / 2, high_vals, bar_w, label="HIGH-diversity calibration (6 diverse prompts)", color="#2a9d8f")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" + ", "\n+ ") for m in methods], fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if baseline is not None:
            ax.axhline(baseline, linestyle="--", color="gray", alpha=0.5, linewidth=1, label=f"reference ({baseline:.2f})")
        for xi, (lv, hv) in enumerate(zip(low_vals, high_vals)):
            fmt = "+.3f" if "Spearman" in title else ".3f"
            ax.text(xi - bar_w / 2, lv, f"{lv:{fmt}}", ha="center",
                    va="bottom" if lv >= 0 else "top", fontsize=7)
            ax.text(xi + bar_w / 2, hv, f"{hv:{fmt}}", ha="center",
                    va="bottom" if hv >= 0 else "top", fontsize=7)
        ax.legend(loc="best", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Effect of calibration-prompt diversity on proxy fidelity "
        f"(K={SUBSET_K}, {N_SPLITS} random splits, test=held-out originals)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    chart_path = out_dir / "with_diverse.png"
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    print(f"Saved chart to {chart_path}")
    print(f"Saved JSON  to {out_dir/'with_diverse_results.json'}")


if __name__ == "__main__":
    main()
