#!/usr/bin/env python3
"""Generate a polished circle packing visualization."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

data = np.load("circle_packing_data.npz")
centers = data["centers"]
radii = data["radii"]
sum_radii = float(data["sum_radii"][0])
n = len(radii)

# --- Color palette ---
# Deep charcoal background, warm gradient for circles
BG_COLOR = "#1a1a2e"
SQUARE_COLOR = "#16213e"
TEXT_COLOR = "#e0e0e0"

# Custom colormap: teal -> gold -> warm coral
cmap = LinearSegmentedColormap.from_list(
    "packing",
    ["#0f4c75", "#3282b8", "#48c9b0", "#f7dc6f", "#f0b27a", "#e74c3c"],
)

# Map radii to colors (larger = warmer)
norm_radii = (radii - radii.min()) / (radii.max() - radii.min() + 1e-12)
colors = cmap(norm_radii)

# Sort by radius so large circles render first, small on top
order = np.argsort(-radii)

# --- Figure setup ---
fig, ax = plt.subplots(1, 1, figsize=(7, 7.6), facecolor=BG_COLOR)
ax.set_facecolor(SQUARE_COLOR)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal")
ax.axis("off")

# Unit square boundary - subtle glow
square = patches.FancyBboxPatch(
    (0, 0), 1, 1,
    boxstyle="round,pad=0.008",
    linewidth=1.5,
    edgecolor="#3282b880",
    facecolor=SQUARE_COLOR,
    zorder=0,
)
ax.add_patch(square)

# --- Draw circles ---
for i in order:
    cx, cy = centers[i]
    r = radii[i]
    c = colors[i]

    # Shadow
    shadow = plt.Circle(
        (cx + 0.003, cy - 0.003), r,
        color="black", alpha=0.25, zorder=1,
    )
    ax.add_patch(shadow)

    # Main circle with thin semi-transparent edge
    circle = plt.Circle(
        (cx, cy), r,
        facecolor=c, edgecolor=(*c[:3], 0.6),
        linewidth=0.8, zorder=2,
    )
    ax.add_patch(circle)

    # Subtle highlight for depth (smaller inner circle, lighter)
    highlight = plt.Circle(
        (cx - r * 0.15, cy + r * 0.15), r * 0.55,
        facecolor="white", alpha=0.07, zorder=3,
    )
    ax.add_patch(highlight)

# --- Title text ---
ax.text(
    0.5, 1.06,
    f"n = {n}",
    transform=ax.transAxes,
    ha="center", va="bottom",
    fontsize=16, fontweight="light",
    color=TEXT_COLOR, fontfamily="sans-serif",
    alpha=0.85,
)
ax.text(
    0.5, 1.02,
    f"\u03A3r = {sum_radii:.4f}",
    transform=ax.transAxes,
    ha="center", va="bottom",
    fontsize=13, fontweight="light",
    color=TEXT_COLOR, fontfamily="sans-serif",
    alpha=0.6,
)

plt.tight_layout(pad=0.5)
plt.savefig(
    "circle_packing_best.png",
    dpi=300, bbox_inches="tight",
    facecolor=BG_COLOR, edgecolor="none",
)
print("Saved circle_packing_best.png")
