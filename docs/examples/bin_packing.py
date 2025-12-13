from typing import Tuple
import algoforge as af
import numpy as np

# Used to generate random (and not very smart) bin packing instances
def generate_bin_packing(n = 100, seed = 216) -> Tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 20, size=n)
    capacity = 30
    return weights, capacity

# Evaluate function for the bin packing solution
def evaluate(proposed_solution, 
             samples: Tuple[np.ndarray, int]) -> dict[str, int]:
    solution_bins = proposed_solution(samples)
    return -len(solution_bins)

bp_instances = [generate_bin_packing(seed=i) for i in range(10)]

af.configure(
    deep_model='openai/gpt-5',
    shallow_model='openai/gpt-5-mini'
)

# Want to use alphaevolve? Then switch to GEPA?
af.alphaevolve(evaluate, bp_instances, budget_dollars=20)
af.gepa(evaluate, bp_instances, budget_seconds=600)

# Or just to do a try run to see which one is best
af.discover(evaluate, bp_instances)
