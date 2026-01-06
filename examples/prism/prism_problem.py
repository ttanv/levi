"""
PRISM Problem: ML Model Placement Optimization.

This module contains the Model dataclass, test case generation,
and evaluation helpers for the PRISM problem.
"""

from dataclasses import dataclass
import numpy as np

GPU_MEM_SIZE = 80  # GB


@dataclass
class Model:
    """Represents an ML model to be placed on GPUs."""
    model_name: str
    model_size: int   # GB
    req_rate: int     # requests per second
    slo: int          # service level objective (latency target)
    cur_gpu_id: int   # current GPU assignment (can be ignored)


def generate_test_cases(num_tests=50, seed=42):
    """Generate test cases matching the leaderboard evaluator."""
    np.random.seed(seed)
    test_cases = []

    for i in range(num_tests):
        gpu_num = np.random.randint(5, 10)
        models = []
        for j in range(gpu_num * 2):
            model_size = np.random.randint(10, 30)
            req_rate = np.random.randint(1, 10)
            slo = np.random.randint(5, 10)
            models.append(Model(
                model_name=f"model_{j}",
                model_size=model_size,
                req_rate=req_rate,
                slo=slo,
                cur_gpu_id=j
            ))
        test_cases.append((gpu_num, models))

    return test_cases


def calculate_kvpr(placement):
    """Calculate the maximum KVPR across all GPUs."""
    max_kvpr = float('-inf')
    for gpu_id, models in placement.items():
        total_model_size = sum(m.model_size for m in models)
        total_weighted_req = sum(m.req_rate / m.slo for m in models)
        remaining_mem = GPU_MEM_SIZE - total_model_size
        if remaining_mem > 0:
            kvpr = total_weighted_req / remaining_mem
        else:
            kvpr = 1000000  # Penalty for exceeding memory
        max_kvpr = max(max_kvpr, kvpr)
    return max_kvpr


def round_robin_placement(gpu_num, models):
    """Baseline: simple round-robin assignment."""
    placement = {i: [] for i in range(gpu_num)}
    for i, model in enumerate(models):
        placement[i % gpu_num].append(model)
    return placement


def compute_theoretical_optimal(gpu_num, models):
    """Compute theoretical optimal (minimum possible) max KVPR."""
    total_weighted_req = sum(m.req_rate / m.slo for m in models)
    total_model_size = sum(m.model_size for m in models)
    total_available_mem = gpu_num * GPU_MEM_SIZE - total_model_size
    if total_available_mem > 0:
        return total_weighted_req / total_available_mem
    return 0.0


# Pre-generated test cases
TEST_CASES = generate_test_cases(num_tests=50)
