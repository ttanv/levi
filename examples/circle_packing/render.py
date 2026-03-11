#!/usr/bin/env python3
"""Render the best circle packing result in a dark-mode friendly layout."""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "circle_packing_data.npz"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "circle_packing_best.png"

# Extracted from the original published PNG so the redraw preserves the circle palette.
BASE_CIRCLE_COLORS = [
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


DARK_THEME = {
    "figure_bg": "#0b1118",
    "axes_bg": "#111a24",
    "title": "#e2edf7",
    "meta": "#a7b9ca",
    "subtitle": "#8599aa",
    "ticks": "#8ea3b6",
    "spine": "#536678",
    "major_grid": "#263545",
    "minor_grid": "#192431",
}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))


def mix_rgb(
    color_a: tuple[float, float, float],
    color_b: tuple[float, float, float],
    amount: float,
) -> tuple[float, float, float]:
    return tuple((1.0 - amount) * a + amount * b for a, b in zip(color_a, color_b))


def tune_fill_for_dark(hex_color: str) -> tuple[float, float, float]:
    """Retune the light-theme palette so it reads cleanly on a dark background."""
    h, l, s = colorsys.rgb_to_hls(*hex_to_rgb(hex_color))
    s = clamp(s * 1.18 + 0.03, 0.28, 0.72)
    l = clamp(l * 0.92 + 0.015, 0.50, 0.69)
    return colorsys.hls_to_rgb(h, l, s)


def edge_for_dark(fill_rgb: tuple[float, float, float]) -> tuple[float, float, float, float]:
    bg_rgb = hex_to_rgb(DARK_THEME["axes_bg"])
    return mix_rgb(fill_rgb, bg_rgb, 0.33) + (0.96,)


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

    if centers.shape != (len(BASE_CIRCLE_COLORS), 2):
        raise ValueError(f"Expected centers shape ({len(BASE_CIRCLE_COLORS)}, 2), got {centers.shape}")
    if radii.shape != (len(BASE_CIRCLE_COLORS),):
        raise ValueError(f"Expected radii shape ({len(BASE_CIRCLE_COLORS)},), got {radii.shape}")

    circle_colors = [tune_fill_for_dark(color) for color in BASE_CIRCLE_COLORS]

    fig = plt.figure(figsize=(8.0, 8.2), dpi=dpi, facecolor=DARK_THEME["figure_bg"])
    ax = fig.add_axes([0.10, 0.10, 0.84, 0.82], facecolor=DARK_THEME["axes_bg"])
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

    ax.grid(which="major", color=DARK_THEME["major_grid"], linewidth=1.0)
    ax.grid(which="minor", color=DARK_THEME["minor_grid"], linewidth=0.8)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=0.9,
        colors=DARK_THEME["ticks"],
        labelsize=10,
        top=True,
        right=True,
        labeltop=False,
        labelright=False,
    )
    ax.tick_params(axis="both", which="minor", length=0)

    for spine in ax.spines.values():
        spine.set_color(DARK_THEME["spine"])
        spine.set_linewidth(1.6)

    for (x, y), radius, color in sorted(zip(centers, radii, circle_colors), key=lambda item: item[1], reverse=True):
        circle = Circle(
            (float(x), float(y)),
            float(radius),
            facecolor=color,
            edgecolor=edge_for_dark(color),
            linewidth=1.0,
            alpha=0.99,
            zorder=3,
        )
        ax.add_patch(circle)

    fig.text(0.10, 0.965, "Circle Packing", fontsize=19, fontweight="bold", color=DARK_THEME["title"])
    fig.text(
        0.94,
        0.965,
        f"n = {len(radii)}  |  score = {sum_radii:.5f}+",
        ha="right",
        fontsize=12,
        color=DARK_THEME["meta"],
    )
    fig.text(
        0.10,
        0.935,
        "Best feasible layout achieved in the unit square [0, 1] x [0, 1]",
        fontsize=10,
        color=DARK_THEME["subtitle"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    render_packing(args.input.resolve(), args.output.resolve(), args.dpi)


if __name__ == "__main__":
    main()
