"""
Circle packing benchmark aligned to common OpenEvolve/AlphaEvolve-style setup.

Task: place n=26 circles inside the unit square [0,1]x[0,1] and maximize the
sum of radii, while enforcing non-overlap and boundary constraints.
"""

import time
from typing import Any

import numpy as np

N_CIRCLES = 26
EPS = 1e-6

PROBLEM_DESCRIPTION = f"""
# Circle Packing (n=26, unit square)

## Problem
Construct a feasible packing of exactly {N_CIRCLES} circles inside the unit square.

Return:
- `centers`: array-like shape ({N_CIRCLES}, 2)
- `radii`: array-like shape ({N_CIRCLES},)
- `sum_radii`: float (objective)

## Feasibility Constraints
- Each circle i must satisfy:
  - `x_i - r_i >= 0`
  - `x_i + r_i <= 1`
  - `y_i - r_i >= 0`
  - `y_i + r_i <= 1`
- Non-overlap for all i != j:
  - `distance(center_i, center_j) >= r_i + r_j`
- Radii must be finite and non-negative.

## Objective
Maximize `sum_radii`.
"""

FUNCTION_SIGNATURE = f"""
import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    '''
    Construct a packing of {N_CIRCLES} circles in the unit square.

    Returns:
        centers: numpy array of shape ({N_CIRCLES}, 2)
        radii: numpy array of shape ({N_CIRCLES},)
        sum_radii: objective value (sum of radii)
    '''
    pass
"""

SEED_PROGRAM = '''import numpy as np

N_CIRCLES = 26


def _compute_max_radii(centers):
    n = centers.shape[0]
    radii = np.zeros(n, dtype=float)
    for i in range(n):
        x, y = centers[i]
        # Boundary limit
        max_r = min(x, y, 1.0 - x, 1.0 - y)

        # Pairwise distance limit
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(centers[i] - centers[j])
            max_r = min(max_r, dist - radii[j])
        radii[i] = max(max_r, 0.0)
    return radii


def run_packing():
    # Simple deterministic seed layout:
    # 1 center circle + 2 rings + 1 corner-ish circle
    centers = np.zeros((N_CIRCLES, 2), dtype=float)
    centers[0] = [0.5, 0.5]

    idx = 1
    for k in range(12):
        angle = 2.0 * np.pi * k / 12.0
        centers[idx] = [0.5 + 0.18 * np.cos(angle), 0.5 + 0.18 * np.sin(angle)]
        idx += 1

    for k in range(12):
        angle = 2.0 * np.pi * k / 12.0
        centers[idx] = [0.5 + 0.34 * np.cos(angle), 0.5 + 0.34 * np.sin(angle)]
        idx += 1

    centers[25] = [0.2, 0.2]

    radii = _compute_max_radii(centers)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
'''


def _to_array(data: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def evaluate_packing_output(centers: np.ndarray, radii: np.ndarray) -> tuple[bool, str]:
    """Return (is_valid, reason)."""
    if np.any(radii < 0):
        return False, "Negative radius"

    # Boundary constraints
    x = centers[:, 0]
    y = centers[:, 1]
    if np.any(x - radii < -EPS) or np.any(x + radii > 1.0 + EPS):
        return False, "Boundary violation on x"
    if np.any(y - radii < -EPS) or np.any(y + radii > 1.0 + EPS):
        return False, "Boundary violation on y"

    # Non-overlap constraints
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dist = float(np.linalg.norm(centers[i] - centers[j]))
            if dist + EPS < float(radii[i] + radii[j]):
                return False, f"Overlap between circles {i} and {j}"

    return True, ""


def compute_behavior_descriptors(centers: np.ndarray, radii: np.ndarray) -> dict[str, float]:
    """Compute geometry descriptors used for behavioral diversity."""
    # Fraction of circles touching at least one boundary.
    x = centers[:, 0]
    y = centers[:, 1]
    margins = np.stack([x - radii, y - radii, 1.0 - x - radii, 1.0 - y - radii], axis=1)
    min_margin = np.min(margins, axis=1)
    boundary_touch_fraction = float(np.mean(min_margin <= 1e-4))

    # Pairwise geometric gaps after subtracting required non-overlap distances.
    pairwise_dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(pairwise_dist, np.inf)
    required_dist = radii[:, None] + radii[None, :]
    gaps = pairwise_dist - required_dist
    np.fill_diagonal(gaps, np.inf)

    nn_gap = np.min(gaps, axis=1)
    nn_gap_mean = float(np.mean(nn_gap))
    nn_gap_std = float(np.std(nn_gap))
    nn_gap_cv = float(nn_gap_std / (abs(nn_gap_mean) + 1e-9))

    # Shannon entropy of normalized radii: high when radii are more uniform.
    radius_sum = float(np.sum(radii))
    if radius_sum <= 0.0:
        radius_entropy = 0.0
    else:
        p = radii / radius_sum
        p = p[p > 0.0]
        if p.size == 0:
            radius_entropy = 0.0
        else:
            radius_entropy = float(-np.sum(p * np.log(p)) / np.log(len(radii)))

    return {
        "boundary_touch_fraction": boundary_touch_fraction,
        "nn_gap_mean": nn_gap_mean,
        "nn_gap_cv": nn_gap_cv,
        "radius_entropy": radius_entropy,
    }


def score_fn(run_packing, _inputs=None) -> dict:
    """
    Score a candidate packing function.

    Returns score = sum_radii for valid packings; otherwise 0.
    """
    try:
        start = time.perf_counter()
        out = run_packing()
        exec_time = time.perf_counter() - start

        if not isinstance(out, tuple) or len(out) != 3:
            return {"error": "run_packing() must return (centers, radii, sum_radii)"}

        centers_raw, radii_raw, sum_radii_raw = out
        centers = _to_array(centers_raw, (N_CIRCLES, 2), "centers")
        radii = _to_array(radii_raw, (N_CIRCLES,), "radii")

        sum_radii = float(sum_radii_raw)
        if not np.isfinite(sum_radii):
            return {"error": "sum_radii is non-finite"}

        descriptors = compute_behavior_descriptors(centers, radii)
        valid, reason = evaluate_packing_output(centers, radii)
        if not valid:
            return {
                "score": 0.0,
                "valid": 0.0,
                "sum_radii": sum_radii,
                "execution_time": exec_time,
                **descriptors,
                "error": reason,
            }

        # Keep objective strict and comparable: maximize sum of radii.
        score = sum_radii
        return {
            "score": score,
            "valid": 1.0,
            "sum_radii": sum_radii,
            "execution_time": exec_time,
            **descriptors,
        }
    except Exception as e:
        return {"error": str(e)}
