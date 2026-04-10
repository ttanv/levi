#!/usr/bin/env python3
"""Plot best-so-far train MSE for a GHG run."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


def _load_problem_module(repo_root: Path):
    problem_path = repo_root / "examples" / "discobench_greenhouse_gas_single_slice" / "problem.py"
    spec = importlib.util.spec_from_file_location("ghg_problem_plot", problem_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load problem module from {problem_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_metrics(problem_module):
    namespace = {
        "__name__": "__candidate__",
        "__source_code__": problem_module.SEED_PROGRAM,
    }
    exec(problem_module.SEED_PROGRAM, namespace)
    fn = namespace[problem_module.TARGET_FUNCTION_NAME]
    return problem_module.score_fn(fn, list(problem_module.TRAIN_DATASETS))


def _format_label(labels: list[str]) -> str:
    if labels == ["Seed"]:
        return "Seed"
    evals = [label.split()[-1] for label in labels]
    if len(evals) == 1:
        return f"Eval {evals[0]}"
    return "Evals " + ", ".join(evals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--max-eval", type=int)
    parser.add_argument(
        "--model-label",
        default="OpenRouter Qwen 30B",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/plots/ghg_best_mse_full.png"),
    )
    args = parser.parse_args()

    snapshot = json.loads(args.snapshot.read_text())
    max_eval = args.max_eval or max(item["eval_number"] for item in snapshot["score_history"])
    repo_root = Path(__file__).resolve().parents[2]
    problem = _load_problem_module(repo_root)
    seed = _seed_metrics(problem)

    elite_by_score = {}
    for elite in snapshot["elites"]:
        elite_by_score.setdefault(elite["primary_score"], elite)

    points = [
        {
            "eval_number": 1,
            "timestamp": next(item["timestamp"] for item in snapshot["score_history"] if item["eval_number"] == 1),
            "best_score": seed["score"],
            "mean_test_mse": seed["mean_test_mse"],
            "labels": ["Seed"],
            "is_endpoint": False,
        }
    ]

    prev = float("-inf")
    for item in snapshot["score_history"]:
        if item["eval_number"] > max_eval:
            break
        if item["best_score"] > prev + 1e-15:
            prev = item["best_score"]
            if item["eval_number"] == 1:
                continue
            elite = elite_by_score[item["best_score"]]
            points.append(
                {
                    "eval_number": item["eval_number"],
                    "timestamp": item["timestamp"],
                    "best_score": item["best_score"],
                    "mean_test_mse": elite["scores"]["mean_test_mse"],
                    "labels": [f"Eval {item['eval_number']}"],
                    "is_endpoint": False,
                }
            )

    last_point = points[-1]
    if last_point["eval_number"] < max_eval:
        points.append(
            {
                "eval_number": max_eval,
                "timestamp": next(
                    (item["timestamp"] for item in snapshot["score_history"] if item["eval_number"] == max_eval),
                    last_point["timestamp"],
                ),
                "best_score": last_point["best_score"],
                "mean_test_mse": last_point["mean_test_mse"],
                "labels": [f"Eval {max_eval}"],
                "is_endpoint": True,
            }
        )

    display_points: list[dict[str, object]] = []
    for point in points:
        if display_points:
            prev_point = display_points[-1]
            prev_mse = float(prev_point["mean_test_mse"])
            curr_mse = float(point["mean_test_mse"])
            if (
                not bool(point.get("is_endpoint"))
                and abs(prev_mse - curr_mse) <= max(1e-9, prev_mse * 1e-12)
            ):
                prev_point["labels"] = list(prev_point["labels"]) + list(point["labels"])
                continue
        display_points.append(dict(point))

    x = [int(point["eval_number"]) for point in points]
    y = [float(point["mean_test_mse"]) for point in points]

    y_min = min(y)
    y_max = max(y)
    y_lower = max(y_min / 1.18, 1e-3)
    y_upper = y_max * 1.18

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12.4, 7.2), dpi=180)
    fig.patch.set_facecolor("#f6f7fb")
    ax.set_facecolor("#ffffff")

    ax.step(x, y, where="post", color="#0f766e", linewidth=3.0, label="Best-so-far mean test MSE")
    ax.fill_between(x, y, [y_upper] * len(y), step="post", color="#99f6e4", alpha=0.18)
    ax.scatter(x[:-1], y[:-1], s=62, color="#0f766e", edgecolor="#ffffff", linewidth=1.3, zorder=3)

    label_offsets = {
        1: (12, -30),
        2: (12, -28),
        30: (12, 12),
        312: (-84, 12),
        401: (-84, -28),
    }
    for point in display_points[:-1]:
        x_offset, y_offset = label_offsets.get(int(point["eval_number"]), (10, -28))
        ax.annotate(
            f"{_format_label(list(point['labels']))}\nMSE {float(point['mean_test_mse']):.3f}",
            (float(point["eval_number"]), float(point["mean_test_mse"])),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=9.2,
            color="#0f172a",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#ffffff",
                "edgecolor": "#cbd5e1",
                "linewidth": 0.8,
                "alpha": 0.96,
            },
        )

    ax.set_title("Greenhouse Gas Model Run: Full 450-Eval Trajectory", fontsize=18, color="#0f172a", pad=18, weight="bold")
    ax.text(
        0.0,
        1.02,
        f"Best-so-far train objective shown as actual mean test MSE across CH4 and SF6 | Model: {args.model_label}",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#475569",
    )
    ax.set_xlabel("Evaluation number", fontsize=12, color="#0f172a")
    ax.set_ylabel("Mean Test MSE", fontsize=12, color="#0f172a")
    ax.set_xlim(0, max_eval + 1)
    ax.set_xticks([0, 100, 200, 300, 400, max_eval])
    ax.set_ylim(y_lower, y_upper)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}" if value >= 100 else f"{value:.1f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, axis="y", color="#e2e8f0", linewidth=1.0)
    ax.grid(False, axis="x")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="both", colors="#334155", labelsize=10)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.12)

    ax.legend(frameon=False, loc="upper right", fontsize=10)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
