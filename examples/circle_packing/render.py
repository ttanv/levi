#!/usr/bin/env python3
"""Render the archived circle packing result with a tighter, axis-based layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "circle_packing_data.npz"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "circle_packing_best.png"

# Extracted from the original published PNG so the redraw preserves the circle palette.
CIRCLE_COLORS = [
    "#84d5ae",
    "#78c7cc",
    "#76c6c4",
    "#6dc4a1",
    "#82c8e3",
    "#91b3e4",
    "#8ec8e5",
    "#72c5b6",
    "#a8e6cf",
    "#74c9a5",
    "#91dcb9",
    "#95c7e5",
    "#89abe3",
    "#6dc4a1",
    "#71c8a4",
    "#98b9e5",
    "#84c8e4",
    "#71c5b2",
    "#6dc5a2",
    "#74caa5",
    "#84d5ae",
    "#7ac7d3",
    "#75c6c1",
    "#80c8e3",
    "#a8e6cf",
    "#6dc5a2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Input .npz with centers/radii")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output image path")
    parser.add_argument("--dpi", type=int, default=250, help="Rendered image DPI")
    return parser.parse_args()


def render_packing(data_path: Path, output_path: Path, dpi: int) -> None:
    data = np.load(data_path)
    centers = np.asarray(data["centers"], dtype=float)
    radii = np.asarray(data["radii"], dtype=float)
    sum_radii = float(np.asarray(data["sum_radii"], dtype=float).reshape(-1)[0])

    if centers.shape != (len(CIRCLE_COLORS), 2):
        raise ValueError(f"Expected centers shape ({len(CIRCLE_COLORS)}, 2), got {centers.shape}")
    if radii.shape != (len(CIRCLE_COLORS),):
        raise ValueError(f"Expected radii shape ({len(CIRCLE_COLORS)},), got {radii.shape}")

    fig = plt.figure(figsize=(8.0, 8.2), dpi=dpi, facecolor="#f3f0ea")
    ax = fig.add_axes([0.10, 0.10, 0.84, 0.82], facecolor="#fbf8f2")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axisbelow(True)

    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.grid(which="major", color="#d8d2c6", linewidth=1.0)
    ax.grid(which="minor", color="#ece7dd", linewidth=0.8)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=0.9,
        colors="#68727a",
        labelsize=10,
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
    )
    ax.tick_params(axis="both", which="minor", length=0)

    for spine in ax.spines.values():
        spine.set_color("#ada79d")
        spine.set_linewidth(1.6)

    for (x, y), radius, color in sorted(zip(centers, radii, CIRCLE_COLORS), key=lambda item: item[1], reverse=True):
        circle = Circle(
            (float(x), float(y)),
            float(radius),
            facecolor=color,
            edgecolor=(1.0, 1.0, 1.0, 0.68),
            linewidth=1.2,
            alpha=0.98,
            zorder=3,
        )
        ax.add_patch(circle)

    ax.set_xlabel("x", fontsize=11, color="#5f6972", labelpad=10)
    ax.set_ylabel("y", fontsize=11, color="#5f6972", labelpad=10)

    fig.text(0.10, 0.965, "Circle Packing", fontsize=19, fontweight="bold", color="#41596e")
    fig.text(0.94, 0.965, f"n = {len(radii)}  |  score = {sum_radii:.5f}+", ha="right", fontsize=12, color="#5e6870")
    fig.text(0.10, 0.935, "Archived best feasible layout in the unit square [0, 1] x [0, 1]", fontsize=10, color="#7b756a")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    render_packing(args.input.resolve(), args.output.resolve(), args.dpi)


if __name__ == "__main__":
    main()
