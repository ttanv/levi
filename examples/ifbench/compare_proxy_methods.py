#!/usr/bin/env python3
"""Offline bake-off of proxy-subset selection methods on IFBench.

Uses two existing IFBench init runs (each 9 prompts × 150 problems). For each
method we calibrate on Run A, then evaluate proxy fidelity on Run B's 9 prompts.
We swap and report the mean across both directions.

Run:
    python examples/ifbench/compare_proxy_methods.py
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
N_RANDOM_SEEDS = 200
RIDGE_ALPHA = 1.0


def load_matrix(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    with open(Path(run_dir) / "proxy_benchmark.json") as f:
        pb = json.load(f)
    rows = [p["loose_instruction_fractions"] for p in pb["prompts"]]
    full = np.array([p["full_score"] for p in pb["prompts"]], dtype=float)
    matrix = np.asarray(rows, dtype=float)
    return matrix, full


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> tuple[np.ndarray, float]:
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    w = np.linalg.solve(A, Xc.T @ yc)
    b = float(y.mean() - X.mean(axis=0) @ w)
    return w, b


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


def select_current(train_matrix: np.ndarray, k: int) -> list[int]:
    sel = select_proxy_problem_subset(train_matrix, k)
    return sel.selected_indices


def select_random(train_matrix: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    return rng.choice(train_matrix.shape[1], size=k, replace=False).tolist()


def select_high_variance(train_matrix: np.ndarray, k: int) -> list[int]:
    stds = train_matrix.std(axis=0)
    return np.argsort(stds)[::-1][:k].tolist()


def select_lowrank_css(train_matrix: np.ndarray, k: int) -> list[int]:
    """Greedy column subset selection: pick columns that span column space best."""
    M = train_matrix.copy()
    R = M.copy()
    selected: list[int] = []
    for _ in range(k):
        norms = np.linalg.norm(R, axis=0)
        for s in selected:
            norms[s] = -np.inf
        idx = int(np.argmax(norms))
        selected.append(idx)
        C = M[:, selected]
        proj = C @ np.linalg.pinv(C) @ M
        R = M - proj
    return selected


def select_kmedoids(train_matrix: np.ndarray, k: int, rng: np.random.Generator) -> tuple[list[int], np.ndarray]:
    """k-medoids on per-problem score vectors (problems live in R^n_prompts)."""
    points = train_matrix.T  # (n_problems, n_prompts)
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
        # average rank for ties
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            sums = np.zeros(len(counts))
            for i, r in zip(inv, ranks):
                sums[i] += r
            avg = sums / counts
            ranks = avg[inv]
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


@dataclass
class MethodResult:
    name: str
    pairwise: float
    spearman: float
    top1_regret: float
    top3_overlap: float


def evaluate(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float, float]:
    return (
        pairwise_order_accuracy(pred, true),
        spearman(pred, true),
        top1_regret(pred, true),
        topk_overlap(pred, true, k=3),
    )


def run_method(
    name: str,
    train_M: np.ndarray, train_full: np.ndarray,
    test_M: np.ndarray, test_full: np.ndarray,
    *,
    select_fn,
    use_ridge: bool,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    indices = select_fn()
    if isinstance(indices, tuple):
        indices, weights = indices
    train_subset = train_M[:, indices]
    test_subset = test_M[:, indices]

    if use_ridge:
        w, b = ridge_fit(train_subset, train_full)
        pred = predict(test_subset, w, b)
    elif weights is not None:
        pred = test_subset @ weights
    else:
        pred = test_subset.mean(axis=1)
    return evaluate(pred, test_full)


def average_directions(*runs: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    arr = np.array(runs)
    return tuple(arr.mean(axis=0).tolist())


def evaluate_method_both_dirs(
    name: str,
    A_M: np.ndarray, A_full: np.ndarray,
    B_M: np.ndarray, B_full: np.ndarray,
    *,
    selector_factory,
    use_ridge: bool,
    n_seeds: int = 1,
) -> MethodResult:
    """selector_factory(train_matrix, rng) -> (indices, optional_weights)."""
    metrics_AB, metrics_BA = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)

        sel_A = selector_factory(A_M, rng)
        weights_A = None
        if isinstance(sel_A, tuple):
            sel_A, weights_A = sel_A
        train_subset_A = A_M[:, sel_A]
        test_subset_AB = B_M[:, sel_A]
        if use_ridge:
            w, b = ridge_fit(train_subset_A, A_full)
            pred_AB = test_subset_AB @ w + b
        elif weights_A is not None:
            pred_AB = test_subset_AB @ weights_A
        else:
            pred_AB = test_subset_AB.mean(axis=1)
        metrics_AB.append(evaluate(pred_AB, B_full))

        rng2 = np.random.default_rng(seed + 10_000)
        sel_B = selector_factory(B_M, rng2)
        weights_B = None
        if isinstance(sel_B, tuple):
            sel_B, weights_B = sel_B
        train_subset_B = B_M[:, sel_B]
        test_subset_BA = A_M[:, sel_B]
        if use_ridge:
            w, b = ridge_fit(train_subset_B, B_full)
            pred_BA = test_subset_BA @ w + b
        elif weights_B is not None:
            pred_BA = test_subset_BA @ weights_B
        else:
            pred_BA = test_subset_BA.mean(axis=1)
        metrics_BA.append(evaluate(pred_BA, A_full))

    all_metrics = np.array(metrics_AB + metrics_BA)
    pairwise, spear, regret, overlap = all_metrics.mean(axis=0).tolist()
    return MethodResult(name, pairwise, spear, regret, overlap)


def main() -> None:
    A_M, A_full = load_matrix(RUN_A)
    B_M, B_full = load_matrix(RUN_B)
    print(f"Run A ({RUN_A}): {A_M.shape}, full ∈ [{A_full.min():.4f}, {A_full.max():.4f}]")
    print(f"Run B ({RUN_B}): {B_M.shape}, full ∈ [{B_full.min():.4f}, {B_full.max():.4f}]")
    print(f"Subset size K = {SUBSET_K}, Ridge α = {RIDGE_ALPHA}, random seeds = {N_RANDOM_SEEDS}")
    print()

    results: list[MethodResult] = []

    # 1. Current method (deterministic) + mean
    results.append(evaluate_method_both_dirs(
        "Current (sep+rank-redund) + mean", A_M, A_full, B_M, B_full,
        selector_factory=lambda M, rng: select_current(M, SUBSET_K),
        use_ridge=False, n_seeds=1,
    ))

    # 2. Current + Ridge (ablation: does Ridge help even with current selector?)
    results.append(evaluate_method_both_dirs(
        "Current + Ridge", A_M, A_full, B_M, B_full,
        selector_factory=lambda M, rng: select_current(M, SUBSET_K),
        use_ridge=True, n_seeds=1,
    ))

    # 3. Random + mean (the dumbest baseline — does any selection beat random?)
    results.append(evaluate_method_both_dirs(
        "Random + mean", A_M, A_full, B_M, B_full,
        selector_factory=lambda M, rng: select_random(M, SUBSET_K, rng),
        use_ridge=False, n_seeds=N_RANDOM_SEEDS,
    ))

    # 4. Random + Ridge (Polo et al. winner)
    results.append(evaluate_method_both_dirs(
        "Random + Ridge (Polo)", A_M, A_full, B_M, B_full,
        selector_factory=lambda M, rng: select_random(M, SUBSET_K, rng),
        use_ridge=True, n_seeds=N_RANDOM_SEEDS,
    ))

    # 5. Low-rank CSS + Ridge
    results.append(evaluate_method_both_dirs(
        "Low-rank CSS + Ridge", A_M, A_full, B_M, B_full,
        selector_factory=lambda M, rng: select_lowrank_css(M, SUBSET_K),
        use_ridge=True, n_seeds=1,
    ))

    # 6. High-variance + Ridge (cheap heuristic)
    results.append(evaluate_method_both_dirs(
        "High-variance + Ridge", A_M, A_full, B_M, B_full,
        selector_factory=lambda M, rng: select_high_variance(M, SUBSET_K),
        use_ridge=True, n_seeds=1,
    ))

    # 7. k-medoids + cluster-weighted mean
    results.append(evaluate_method_both_dirs(
        "k-medoids + weighted mean", A_M, A_full, B_M, B_full,
        selector_factory=lambda M, rng: select_kmedoids(M, SUBSET_K, rng),
        use_ridge=False, n_seeds=N_RANDOM_SEEDS,
    ))

    print_table(results)
    print()
    print_takeaways(results)
    save_artifacts(results, A_full, B_full)


def save_artifacts(results: list[MethodResult], A_full: np.ndarray, B_full: np.ndarray) -> None:
    out_dir = Path("runs/proxy_method_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "subset_k": SUBSET_K,
        "ridge_alpha": RIDGE_ALPHA,
        "n_random_seeds": N_RANDOM_SEEDS,
        "run_A": RUN_A,
        "run_B": RUN_B,
        "run_A_full_range": [float(A_full.min()), float(A_full.max())],
        "run_B_full_range": [float(B_full.min()), float(B_full.max())],
        "results": [
            {
                "name": r.name,
                "pairwise": r.pairwise,
                "spearman": r.spearman,
                "top1_regret": r.top1_regret,
                "top3_overlap": r.top3_overlap,
            }
            for r in results
        ],
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  Saved JSON to {out_dir/'results.json'} (matplotlib not available; skipping chart)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_specs = [
        ("Pairwise order accuracy (↑)", [r.pairwise for r in results], 0.5, "higher is better — 0.5 = random"),
        ("Spearman ρ (↑)", [r.spearman for r in results], 0.0, "higher is better — 0.0 = no rank correlation"),
        ("Top-1 regret (↓)", [r.top1_regret for r in results], None, "lower is better — gap from picking wrong #1"),
        ("Top-3 overlap (↑)", [r.top3_overlap for r in results], 3/9, "higher is better — 0.33 = random 3-of-9"),
    ]
    names = [r.name for r in results]
    colors = ["#888888" if n.startswith("Current") else "#4477aa" for n in names]
    short_names = [n.replace(" + ", "\n+ ") for n in names]

    for ax, (title, vals, baseline, subtitle) in zip(axes.flat, metric_specs):
        bars = ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(short_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(subtitle, fontsize=8, color="#666")
        if baseline is not None:
            ax.axvline(baseline, linestyle="--", color="red", alpha=0.5, linewidth=1)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {val:+.3f}" if title.startswith("Spearman") else f" {val:.3f}",
                    va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Proxy-subset method bake-off (K={SUBSET_K} problems, IFBench discovery=150)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    chart_path = out_dir / "comparison.png"
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    print(f"  Saved chart to {chart_path}")
    print(f"  Saved JSON  to {out_dir/'results.json'}")


def print_table(results: list[MethodResult]) -> None:
    headers = ["Method", "Pairwise↑", "Spearman↑", "Top-1 regret↓", "Top-3 overlap↑"]
    rows = [
        [r.name, f"{r.pairwise:.3f}", f"{r.spearman:+.3f}", f"{r.top1_regret:.4f}", f"{r.top3_overlap:.3f}"]
        for r in results
    ]
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "  ".join("─" * w for w in widths)

    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(sep)
    for r in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))


def print_takeaways(results: list[MethodResult]) -> None:
    print("─" * 70)
    print("Takeaways:")
    by_pairwise = sorted(results, key=lambda r: -r.pairwise)
    by_regret = sorted(results, key=lambda r: r.top1_regret)
    by_spearman = sorted(results, key=lambda r: -r.spearman)
    print(f"  Best pairwise order accuracy: {by_pairwise[0].name} ({by_pairwise[0].pairwise:.3f})")
    print(f"  Best Spearman ρ:              {by_spearman[0].name} ({by_spearman[0].spearman:+.3f})")
    print(f"  Lowest top-1 regret:          {by_regret[0].name} ({by_regret[0].top1_regret:.4f})")
    current = next(r for r in results if r.name.startswith("Current"))
    random_ridge = next(r for r in results if "Random + Ridge" in r.name)
    print()
    print(f"  Current method pairwise:      {current.pairwise:.3f}")
    print(f"  Random+Ridge baseline:        {random_ridge.pairwise:.3f}  "
          f"(Δ = {random_ridge.pairwise - current.pairwise:+.3f})")


if __name__ == "__main__":
    main()
