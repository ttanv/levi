"""Tests for proxy benchmark discovery helpers."""

import numpy as np
import pytest

from levi.init.proxy_benchmark import (
    build_problem_score_matrix,
    select_inputs_by_index,
    select_proxy_problem_subset,
)


def test_build_problem_score_matrix_supports_non_binary_scores():
    matrix = build_problem_score_matrix(
        [
            {"problem_scores": [0.0, 0.5, 1.0]},
            {"problem_scores": [1.0, 0.25, 0.75]},
        ],
        key="problem_scores",
    )

    assert matrix.shape == (2, 3)
    assert np.allclose(matrix[0], [0.0, 0.5, 1.0])
    assert np.allclose(matrix[1], [1.0, 0.25, 0.75])


def test_build_problem_score_matrix_rejects_inconsistent_widths():
    with pytest.raises(ValueError, match="expected 2"):
        build_problem_score_matrix(
            [
                {"problem_scores": [0.0, 1.0]},
                {"problem_scores": [1.0]},
            ],
            key="problem_scores",
        )


def test_selector_avoids_redundant_columns_when_possible():
    matrix = np.asarray(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    selection = select_proxy_problem_subset(matrix, subset_size=2)

    assert len(selection.selected_indices) == 2
    assert not ({0, 1} <= set(selection.selected_indices))


def test_selector_prefers_columns_that_more_cleanly_separate_prompts():
    matrix = np.asarray(
        [
            [0.9, 0.9],
            [0.8, 0.6],
            [0.7, 0.3],
            [0.6, 0.0],
        ],
        dtype=float,
    )

    selection = select_proxy_problem_subset(matrix, subset_size=1)

    assert selection.selected_indices == [1]


def test_select_inputs_by_index_preserves_requested_order():
    inputs = ["p0", "p1", "p2", "p3"]

    assert select_inputs_by_index(inputs, [2, 0, 3]) == ["p2", "p0", "p3"]
