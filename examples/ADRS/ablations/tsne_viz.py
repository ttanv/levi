#!/usr/bin/env python3
"""
t-SNE visualization of elite program diversity: noise vs no_noise
Uses TF-IDF on code tokens as embeddings.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

# microsoft/codebert-base — dedicated code embedding model (RoBERTa trained on code)
print("Loading CodeBERT embedding model...")
EMBED_MODEL = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
print("Model loaded.")

BASE = Path("runs/ablations/anova_features")

def load_elites(problem: str, condition: str, label: str):
    """Load all elite codes from all seeds of a condition."""
    records = []
    cond_map = {
        "txn": {"noise": "pca_noise", "no_noise": "pca_no_noise"},
        "cbl": {"noise": "anova_noise", "no_noise": "anova_no_noise"},
    }
    folder_cond = cond_map[problem][condition]
    for seed in range(3):
        snap = BASE / problem / f"{folder_cond}_seed{seed}" / "snapshot.json"
        if not snap.exists():
            continue
        s = json.load(open(snap))
        for e in s.get("elites", []):
            records.append({
                "code": e["code"],
                "score": e["primary_score"],
                "label": label,
                "seed": seed,
                "problem": problem,
                "condition": condition,
            })
    return records



def plot_tsne(records, title, out_path, perplexity=30):
    codes = [r["code"] for r in records]
    labels = [r["label"] for r in records]
    scores = [r["score"] for r in records]
    seeds = [r["seed"] for r in records]

    # CodeT5+ embeddings (truncate to 512 chars to fit context window)
    print(f"  Embedding {len(codes)} programs...")
    X = EMBED_MODEL.encode(codes, batch_size=16, show_progress_bar=True)

    # Reduce to 50 dims with PCA first (speeds up t-SNE, removes noise)
    n_components = min(50, X.shape[1], X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)

    # t-SNE
    perp = min(perplexity, len(records) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_reduced)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    color_map = {"noise": "#e74c3c", "no_noise": "#3498db"}
    marker_map = {0: "o", 1: "s", 2: "^"}

    unique_labels = sorted(set(labels))

    # Left plot: colored by condition
    ax = axes[0]
    for lbl in unique_labels:
        mask = [i for i, l in enumerate(labels) if l == lbl]
        condition = records[mask[0]]["condition"]
        color = color_map[condition]
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color, alpha=0.7, s=60, edgecolors='white', linewidths=0.5,
            label=lbl
        )
    ax.set_title("Colored by condition")
    ax.legend(fontsize=9)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)

    # Right plot: colored by score
    ax = axes[1]
    all_scores = np.array(scores)
    sc = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=all_scores, cmap='viridis', alpha=0.8, s=60,
        edgecolors='white', linewidths=0.5
    )
    # Mark noise vs no_noise with different shapes
    for lbl in unique_labels:
        mask = [i for i, l in enumerate(labels) if l == lbl]
        condition = records[mask[0]]["condition"]
        marker = "o" if condition == "noise" else "s"
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=all_scores[mask], cmap='viridis', alpha=0.8, s=70,
            marker=marker, edgecolors='white', linewidths=0.5
        )
    plt.colorbar(sc, ax=ax, label="Score")
    noise_patch = mpatches.Patch(color='gray', label='circle=noise, square=no_noise')
    ax.legend(handles=[noise_patch], fontsize=8)
    ax.set_title("Colored by score")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")

    # Print spread stats
    print(f"\nSpread statistics ({title}):")
    for lbl in unique_labels:
        mask = [i for i, l in enumerate(labels) if l == lbl]
        pts = X_2d[mask]
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        condition = records[mask[0]]["condition"]
        mean_score = np.mean([scores[i] for i in mask])
        print(f"  {lbl:20s}: n={len(mask):3d}, spread={dists.mean():.2f}±{dists.std():.2f}, mean_score={mean_score:.1f}")

    return X_2d


out_dir = Path("runs/ablations/anova_features/plots")
out_dir.mkdir(parents=True, exist_ok=True)

# TXN
print("Loading TXN elites...")
txn_records = (
    load_elites("txn", "noise",    "TXN noise")    +
    load_elites("txn", "no_noise", "TXN no_noise")
)
print(f"  {len(txn_records)} total elites")
if txn_records:
    plot_tsne(txn_records, "TXN Scheduling — Elite Diversity (noise vs no_noise)",
              out_dir / "tsne_txn.png")

# CBL
print("\nLoading CBL elites...")
cbl_records = (
    load_elites("cbl", "noise",    "CBL noise")    +
    load_elites("cbl", "no_noise", "CBL no_noise")
)
print(f"  {len(cbl_records)} total elites")
if cbl_records:
    plot_tsne(cbl_records, "CBL (Can't Be Late) — Elite Diversity (noise vs no_noise)",
              out_dir / "tsne_cbl.png")

# Combined
print("\nCombined plot (all 4 conditions)...")
all_records = txn_records + cbl_records
if all_records:
    codes = [r["code"] for r in all_records]
    print(f"  Embedding {len(codes)} programs for combined plot...")
    X = EMBED_MODEL.encode(codes, batch_size=16, show_progress_bar=True)
    n_components = min(50, X.shape[1], X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    perp = min(40, len(all_records) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_reduced)

    fig, ax = plt.subplots(figsize=(10, 8))
    color_cond = {"noise": "#e74c3c", "no_noise": "#3498db"}
    marker_prob = {"txn": "o", "cbl": "s"}
    for r, (x, y) in zip(all_records, X_2d):
        ax.scatter(x, y,
                   c=color_cond[r["condition"]],
                   marker=marker_prob[r["problem"]],
                   alpha=0.7, s=55, edgecolors='white', linewidths=0.4)

    legend_elems = [
        mpatches.Patch(color="#e74c3c", label="noise=0.2"),
        mpatches.Patch(color="#3498db", label="no_noise"),
        plt.Line2D([0],[0], marker='o', color='gray', label='TXN', linestyle='None', markersize=8),
        plt.Line2D([0],[0], marker='s', color='gray', label='CBL', linestyle='None', markersize=8),
    ]
    ax.legend(handles=legend_elems, fontsize=10)
    ax.set_title("All elites: TXN + CBL, noise vs no_noise", fontsize=13, fontweight='bold')
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "tsne_combined.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {out_dir}/tsne_combined.png")
