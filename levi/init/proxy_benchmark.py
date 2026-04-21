"""Helpers for learning a smaller proxy benchmark from init-stage evaluations."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProxyCandidateBreakdown:
    """Objective breakdown for one candidate problem at one greedy step."""

    problem_index: int
    objective: float
    separation_score: float
    ranking_score: float
    redundancy_penalty: float


@dataclass(frozen=True)
class ProxySelectionStep:
    """Trace for one greedy forward-selection step."""

    step_number: int
    selected_so_far: list[int]
    chosen: ProxyCandidateBreakdown
    top_candidates: list[ProxyCandidateBreakdown]


@dataclass(frozen=True)
class ProxyBenchmarkSelection:
    """Summary of a greedy representative-problem selection pass."""

    selected_indices: list[int]
    objective_trace: list[float]
    full_prompt_scores: list[float]
    proxy_prompt_scores: list[float]
    final_ranking_score: float
    step_traces: list[ProxySelectionStep]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "selected_indices": list(self.selected_indices),
            "objective_trace": list(self.objective_trace),
            "full_prompt_scores": list(self.full_prompt_scores),
            "proxy_prompt_scores": list(self.proxy_prompt_scores),
            "final_ranking_score": float(self.final_ranking_score),
            "step_traces": [asdict(step) for step in self.step_traces],
        }


def build_problem_score_matrix(results: list[dict[str, Any]], *, key: str) -> np.ndarray:
    """Extract a dense prompt-by-problem score matrix from evaluation results."""
    rows: list[list[float]] = []
    expected_width: int | None = None

    for idx, result in enumerate(results):
        raw_scores = result.get(key)
        if not isinstance(raw_scores, (list, tuple)):
            raise ValueError(f"Result {idx} is missing a list-valued '{key}' entry")

        row = [float(value) for value in raw_scores]
        if expected_width is None:
            expected_width = len(row)
        elif len(row) != expected_width:
            raise ValueError(
                f"Result {idx} has {len(row)} per-problem scores, expected {expected_width}"
            )
        rows.append(row)

    if not rows:
        raise ValueError("Cannot build a proxy benchmark from an empty result set")
    if expected_width is None or expected_width <= 0:
        raise ValueError("Per-problem score vectors must be non-empty")

    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D prompt-by-problem matrix, got shape {matrix.shape}")
    return matrix


def select_proxy_problem_subset(
    score_matrix: np.ndarray,
    subset_size: int,
    *,
    trace_top_k: int = 5,
) -> ProxyBenchmarkSelection:
    """Greedily choose problems whose subset scores best proxy the full benchmark."""
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D score matrix, got shape {matrix.shape}")

    n_prompts, n_problems = matrix.shape
    if n_prompts <= 0 or n_problems <= 0:
        raise ValueError("Score matrix must have at least one prompt and one problem")

    target_size = min(max(int(subset_size), 0), n_problems)
    if target_size <= 0:
        return ProxyBenchmarkSelection(
            selected_indices=[],
            objective_trace=[],
            full_prompt_scores=matrix.mean(axis=1).tolist(),
            proxy_prompt_scores=[],
            final_ranking_score=0.0,
            step_traces=[],
        )

    full_prompt_scores = matrix.mean(axis=1)
    selected: list[int] = []
    objective_trace: list[float] = []
    step_traces: list[ProxySelectionStep] = []
    remaining = list(range(n_problems))

    while remaining and len(selected) < target_size:
        candidate_breakdowns = [
            _candidate_breakdown(matrix, selected + [problem_idx], full_prompt_scores)
            for problem_idx in remaining
        ]
        candidate_breakdowns.sort(key=lambda item: (-item.objective, item.problem_index))
        best = candidate_breakdowns[0] if candidate_breakdowns else None

        if best is None:
            break

        selected.append(best.problem_index)
        remaining.remove(best.problem_index)
        objective_trace.append(best.objective)
        step_traces.append(
            ProxySelectionStep(
                step_number=len(selected),
                selected_so_far=list(selected),
                chosen=best,
                top_candidates=list(candidate_breakdowns[: max(1, trace_top_k)]),
            )
        )

    proxy_prompt_scores = matrix[:, selected].mean(axis=1).tolist() if selected else []
    final_ranking_score = _pairwise_order_accuracy(full_prompt_scores, np.asarray(proxy_prompt_scores, dtype=float))
    return ProxyBenchmarkSelection(
        selected_indices=selected,
        objective_trace=objective_trace,
        full_prompt_scores=full_prompt_scores.tolist(),
        proxy_prompt_scores=proxy_prompt_scores,
        final_ranking_score=final_ranking_score,
        step_traces=step_traces,
    )


def select_inputs_by_index(inputs: list[Any], indices: list[int]) -> list[Any]:
    """Pick a stable subset of problem inputs by index."""
    return [inputs[i] for i in indices]


def _candidate_breakdown(
    matrix: np.ndarray,
    selected: list[int],
    full_prompt_scores: np.ndarray,
) -> ProxyCandidateBreakdown:
    proxy_scores = matrix[:, selected].mean(axis=1)
    separation_score = _mean_problem_separation(matrix, selected)
    ranking_score = _pairwise_order_accuracy(full_prompt_scores, proxy_scores)
    redundancy_penalty = _mean_redundancy(matrix, selected[-1], selected[:-1]) if len(selected) > 1 else 0.0
    objective = (0.5 * separation_score) + (0.5 * ranking_score) - (0.15 * redundancy_penalty)
    return ProxyCandidateBreakdown(
        problem_index=selected[-1],
        objective=objective,
        separation_score=separation_score,
        ranking_score=ranking_score,
        redundancy_penalty=redundancy_penalty,
    )


def _mean_problem_separation(matrix: np.ndarray, selected: list[int]) -> float:
    if not selected:
        return 0.0

    column_spread = np.std(matrix, axis=0)
    max_spread = float(np.max(column_spread))
    if max_spread <= 1e-12:
        return 0.0
    return float(np.mean(column_spread[selected]) / max_spread)


def _pairwise_order_accuracy(full_scores: np.ndarray, proxy_scores: np.ndarray) -> float:
    n = len(full_scores)
    if n <= 1:
        return 1.0

    total = 0
    score = 0.0
    for left in range(n):
        for right in range(left + 1, n):
            full_delta = float(full_scores[left] - full_scores[right])
            proxy_delta = float(proxy_scores[left] - proxy_scores[right])
            total += 1
            if abs(full_delta) <= 1e-12 and abs(proxy_delta) <= 1e-12:
                score += 1.0
            elif abs(full_delta) <= 1e-12 or abs(proxy_delta) <= 1e-12:
                score += 0.5
            elif full_delta * proxy_delta > 0:
                score += 1.0
    return score / total if total else 1.0


def _mean_redundancy(matrix: np.ndarray, candidate_idx: int, selected_indices: list[int]) -> float:
    if not selected_indices:
        return 0.0

    candidate = matrix[:, candidate_idx]
    similarities = [_absolute_correlation(candidate, matrix[:, other_idx]) for other_idx in selected_indices]
    return float(np.mean(similarities)) if similarities else 0.0


def _absolute_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 1e-12 or right_std <= 1e-12:
        return 1.0 if np.allclose(left, right) else 0.0

    corr = np.corrcoef(left, right)[0, 1]
    if np.isnan(corr):
        return 0.0
    return abs(float(corr))
