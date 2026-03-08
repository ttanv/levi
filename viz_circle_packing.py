#!/usr/bin/env python3
"""Generate a clean circle packing visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

data = np.load("circle_packing_data.npz")
centers = data["centers"]
radii = data["radii"]
n = len(radii)

# --- Style: white bg, soft pastel circles, clean text ---
BG_COLOR = "white"
BORDER_COLOR = "#b0b0b0"
TEXT_COLOR = "#555555"

# Pastel teal-to-blue gradient (matching the reference style)
cmap = LinearSegmentedColormap.from_list(
    "packing", ["#a8e6cf", "#88d8b0", "#6cc4a1", "#7ec8e3", "#a7c7e7", "#89abe3"],
)
norm_radii = (radii - radii.min()) / (radii.max() - radii.min() + 1e-12)
colors = cmap(norm_radii)

# --- Figure ---
fig, ax = plt.subplots(1, 1, figsize=(6, 6.8), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(-0.04, 1.04)
ax.set_ylim(-0.04, 1.04)
ax.set_aspect("equal")
ax.axis("off")

# Unit square border - thin solid line
square = plt.Rectangle((0, 0), 1, 1, linewidth=1.2, edgecolor=BORDER_COLOR,
                        facecolor="white", zorder=0)
ax.add_patch(square)

# Corner tick marks (like the reference)
tick = 0.03
for (cx, cy), (dx, dy) in [
    ((0, 0), (1, 1)), ((1, 0), (-1, 1)),
    ((0, 1), (1, -1)), ((1, 1), (-1, -1)),
]:
    ax.plot([cx, cx + dx * tick], [cy, cy], color=BORDER_COLOR, lw=1.2,
            solid_capstyle="round", zorder=1)
    ax.plot([cx, cx], [cy, cy + dy * tick], color=BORDER_COLOR, lw=1.2,
            solid_capstyle="round", zorder=1)

# --- Draw circles (no outlines, no shadows, no highlights) ---
order = np.argsort(-radii)
for i in order:
    circle = plt.Circle(
        centers[i], radii[i],
        facecolor=colors[i], edgecolor="none", zorder=2,
    )
    ax.add_patch(circle)

# --- Text below the square ---
ax.text(0.5, -0.06, f"n = {n}  |  score = 2.63598+",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=11, color=TEXT_COLOR, fontfamily="sans-serif")

plt.tight_layout(pad=0.3)
plt.savefig("circle_packing_best.png", dpi=300, bbox_inches="tight",
            facecolor=BG_COLOR, edgecolor="none")
print("Saved circle_packing_best.png")
