#!/usr/bin/env python3
"""Budget-tradeoff heatmap using k-medoids problem selection + weighted-mean predictor.

For each random split, we cluster problems into K groups via k-medoids on the
calibration prompts' columns, pick cluster medoids, and predict held-out
prompts as a cluster-size-weighted mean over medoid scores (SubLIME-style).

Run:
    python examples/ifbench/budget_tradeoff_kmedoids.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ORIGINAL_RUNS = ["runs/20260419_100536_ifbench", "runs/20260419_223656_ifbench"]
DIVERSE_FILE = "runs/diverse_prompts/diverse_prompts.json"

N_INIT_VALUES = [5, 7, 9, 11, 13, 15]
K_PROXY_VALUES = [10, 15, 20, 25, 30, 35, 40]
N_TRIALS = 60
N_ITER_FOR_COST = 50
N_FULL_PROBLEMS = 150


def load_pool():
    rows, fulls = [], []
    for run in ORIGINAL_RUNS:
        with open(Path(run) / "proxy_benchmark.json") as f:
            pb = json.load(f)
        for p in pb["prompts"]:
            rows.append(p["loose_instruction_fractions"])
            fulls.append(p["full_score"])
    with open(DIVERSE_FILE) as f:
        diverse = json.load(f)
    for p in diverse["prompts"]:
        rows.append(p["loose_instruction_fractions"])
        fulls.append(p["full_score"])
    return np.asarray(rows, dtype=float), np.asarray(fulls, dtype=float)


def spearman(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def top1_regret(true_scores, pred_scores):
    pick = int(np.argmax(pred_scores))
    return float(true_scores.max() - true_scores[pick])


def select_kmedoids(M_cal, k, rng, max_iter=50):
    """k-medoids on problem columns (features = calibration prompt scores)."""
    points = M_cal.T  # shape (n_problems, n_cal_prompts)
    n = points.shape[0]
    medoids = rng.choice(n, size=k, replace=False).tolist()
    for _ in range(max_iter):
        d = np.linalg.norm(points[:, None] - points[medoids][None, :], axis=2)
        assign = np.argmin(d, axis=1)
        new = []
        for c in range(k):
            cluster = np.where(assign == c)[0]
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
    assign = np.argmin(d, axis=1)
    sizes = np.array([(assign == c).sum() for c in range(k)], dtype=float)
    return medoids, sizes / sizes.sum()


def evaluate_cell(M, full, n_init, k_proxy, n_trials, seed):
    rng = np.random.default_rng(seed)
    n_pool = M.shape[0]
    n_problems = M.shape[1]
    k_proxy = min(k_proxy, n_problems)
    n_init = min(n_init, n_pool - 3)
    spearmans, regrets = [], []
    for _ in range(n_trials):
        perm = rng.permutation(n_pool)
        cal_idx = perm[:n_init]
        test_idx = perm[n_init:]
        M_cal = M[cal_idx]
        sel, weights = select_kmedoids(M_cal, k_proxy, rng)
        pred = M[test_idx][:, sel] @ weights
        true = full[test_idx]
        spearmans.append(spearman(pred, true))
        regrets.append(top1_regret(true, pred))
    return float(np.nanmean(spearmans)), float(np.nanmean(regrets))


def main():
    M, full = load_pool()
    print(f"Pool: {M.shape[0]} prompts × {M.shape[1]} problems (k-medoids + weighted)", flush=True)

    spearman_grid = np.full((len(K_PROXY_VALUES), len(N_INIT_VALUES)), np.nan)
    regret_grid = np.full((len(K_PROXY_VALUES), len(N_INIT_VALUES)), np.nan)
    total_cells = len(K_PROXY_VALUES) * len(N_INIT_VALUES)
    cell_i = 0
    t_start = time.time()
    for ki, k in enumerate(K_PROXY_VALUES):
        for ni, n in enumerate(N_INIT_VALUES):
            cell_i += 1
            t0 = time.time()
            sp, reg = evaluate_cell(M, full, n, k, N_TRIALS, seed=ki * 100 + ni)
            spearman_grid[ki, ni] = sp
            regret_grid[ki, ni] = reg
            print(f"  [{cell_i:2d}/{total_cells}] N_init={n:3d} K_proxy={k:3d}  "
                  f"spearman={sp:+.3f}  top1_regret={reg:.4f}  "
                  f"({time.time() - t0:.1f}s, total {time.time() - t_start:.0f}s)",
                  flush=True)

    save_heatmap(spearman_grid, regret_grid)
    print_summary(spearman_grid, regret_grid)


def save_heatmap(spearman_grid, regret_grid):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, grid, title, cmap, fmt in [
        (axes[0], spearman_grid, "Spearman ρ (higher = better ranking)", "viridis", "{:+.2f}"),
        (axes[1], regret_grid, "Top-1 regret (lower = better best-pick)", "viridis_r", "{:.3f}"),
    ]:
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xticks(range(len(N_INIT_VALUES)))
        ax.set_xticklabels(N_INIT_VALUES)
        ax.set_yticks(range(len(K_PROXY_VALUES)))
        ax.set_yticklabels(K_PROXY_VALUES)
        ax.set_xlabel("N_init  (number of calibration prompts)", fontsize=11)
        ax.set_ylabel("K_proxy  (proxy subset size, problems)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for ki in range(len(K_PROXY_VALUES)):
            for ni in range(len(N_INIT_VALUES)):
                val = grid[ki, ni]
                ax.text(ni, ki, fmt.format(val), ha="center", va="center",
                        color="white" if (cmap == "viridis_r") == (val > np.nanmedian(grid)) else "black",
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        Kgrid, Ngrid = np.meshgrid(K_PROXY_VALUES, N_INIT_VALUES, indexing="ij")
        cost = Ngrid * N_FULL_PROBLEMS + N_ITER_FOR_COST * Kgrid
        x = np.arange(len(N_INIT_VALUES))
        y = np.arange(len(K_PROXY_VALUES))
        X, Y = np.meshgrid(x, y)
        levels = sorted({1500, 2500, 3500, 5000, 7500, 10000, 15000, 20000})
        cs = ax.contour(X, Y, cost, levels=levels, colors="white",
                        linewidths=1.2, alpha=0.85, linestyles="--")
        ax.clabel(cs, inline=True, fontsize=8, fmt=lambda v: f"{int(v):,} calls")

        try:
            ni = N_INIT_VALUES.index(9)
            ki = K_PROXY_VALUES.index(15)
            ax.plot(ni, ki, "o", markersize=18, markerfacecolor="none",
                    markeredgecolor="red", markeredgewidth=2.5, label="current (9,15)")
            ax.legend(loc="lower right", fontsize=9)
        except ValueError:
            pass

    fig.suptitle(
        f"k-medoids + weighted-mean (SubLIME-style): ranking quality over (N_init, K_proxy) — "
        f"{N_TRIALS} random splits.  Iso-cost = N_init·150 + {N_ITER_FOR_COST}·K_proxy",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    out = Path("runs/proxy_method_comparison/budget_tradeoff_kmedoids.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out}", flush=True)


def print_summary(spearman_grid, regret_grid):
    print("\nBest cells by Spearman (k-medoids + weighted):", flush=True)
    flat = [(spearman_grid[ki, ni], K_PROXY_VALUES[ki], N_INIT_VALUES[ni],
             N_INIT_VALUES[ni] * N_FULL_PROBLEMS + N_ITER_FOR_COST * K_PROXY_VALUES[ki])
            for ki in range(len(K_PROXY_VALUES))
            for ni in range(len(N_INIT_VALUES))]
    flat.sort(key=lambda x: -x[0])
    print(f"  {'rank':>4s}  {'spearman':>8s}  {'K_proxy':>8s}  {'N_init':>7s}  {'cost (calls)':>12s}")
    for r, (sp, k, n, cost) in enumerate(flat[:8], 1):
        print(f"  {r:>4d}  {sp:>+8.3f}  {k:>8d}  {n:>7d}  {cost:>12,d}")

    print("\nBest at iso-cost bands (within ±15%):")
    for target in [1500, 2500, 3500, 5000, 7500]:
        candidates = [(sp, k, n, c) for sp, k, n, c in flat
                      if abs(c - target) / target <= 0.15]
        if not candidates:
            continue
        sp, k, n, c = max(candidates, key=lambda x: x[0])
        print(f"  ~{target:>5,d} calls: spearman={sp:+.3f} at N_init={n}, K_proxy={k} (cost {c:,})")


if __name__ == "__main__":
    main()
