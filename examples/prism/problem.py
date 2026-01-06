"""
PRISM (ML Model Placement) Problem Definition.

Contains problem description, prompts, scoring function, and test inputs.
"""

from dataclasses import dataclass
import numpy as np

# --- Problem Infrastructure ---
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


# --- Prompts ---
PROBLEM_DESCRIPTION = """
# Model Placement Optimization

## Problem
Assign models to GPUs to minimize maximum KVPR (KV Cache Pressure).

## Key Concepts
- Each model has: model_size (10-30), req_rate (1-10), slo (5-10)
- GPU memory limit: 80 GB
- KVPR = sum(req_rate/slo) / (80 - sum(model_size)) for models on a GPU
- Goal: Minimize the maximum KVPR across all GPUs

## Input
- `gpu_num`: Number of GPUs (5-10)
- `models`: List of model objects. Each model has attributes:
  - `m.model_size` (int)
  - `m.req_rate` (int)
  - `m.slo` (int)

## Output
Return a dict: `{gpu_id: [list of models], ...}`
- Keys are integers 0 to gpu_num-1
- Values are lists containing model objects from the input
- Every model must be assigned to exactly one GPU
- Total model_size per GPU must not exceed 80

## You can import: random, heapq, collections, math, numpy, itertools
"""

FUNCTION_SIGNATURE = """
def compute_model_placement(gpu_num, models):
    '''
    Args:
        gpu_num: Number of GPUs (int)
        models: List of models with .model_size, .req_rate, .slo

    Returns:
        dict: {gpu_id: [models assigned to that GPU]}
    '''
    pass
"""

SEED_PROGRAM = '''def compute_model_placement(gpu_num, models):
    """Greedy placement: assign each model to GPU with lowest current KVPR."""
    # Sort by req_rate/slo descending (high priority first)
    sorted_models = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)

    # Initialize placement and tracking
    placement = {i: [] for i in range(gpu_num)}
    mem_used = [0] * gpu_num      # memory used per GPU
    load = [0.0] * gpu_num        # sum of req_rate/slo per GPU

    for model in sorted_models:
        best_gpu = None
        best_kvpr = float('inf')

        for gpu_id in range(gpu_num):
            # Check memory constraint
            if mem_used[gpu_id] + model.model_size > 80:
                continue

            # Calculate KVPR if we add this model
            new_load = load[gpu_id] + model.req_rate / model.slo
            new_mem = mem_used[gpu_id] + model.model_size
            remaining = 80 - new_mem
            if remaining > 0:
                kvpr = new_load / remaining
                if kvpr < best_kvpr:
                    best_kvpr = kvpr
                    best_gpu = gpu_id

        if best_gpu is None:
            # Fallback: pick GPU with most free memory
            best_gpu = min(range(gpu_num), key=lambda g: mem_used[g])

        placement[best_gpu].append(model)
        mem_used[best_gpu] += model.model_size
        load[best_gpu] += model.req_rate / model.slo

    return placement
'''

SEED_INSPIRATIONS = []

DIVERSITY_SEED_PROMPT = """
# Model Placement Optimization

## Problem
Assign models to GPUs to minimize maximum KVPR.
- KVPR = sum(req_rate/slo) / (80 - sum(model_size)) per GPU
- Memory limit: 80 GB per GPU

## Input
- `gpu_num`: Number of GPUs (5-10)
- `models`: List with .model_size, .req_rate, .slo attributes

## Output
Return dict: `{gpu_id: [models]}`

## Function
```python
def compute_model_placement(gpu_num, models):
    # Returns {0: [...], 1: [...], ...}
    pass
```

## Your Task: ALGORITHMIC DIVERSITY

Design a **FUNDAMENTALLY DIFFERENT ALGORITHM** than existing seeds.

## Existing Seeds:
{existing_seeds}

## Output
Output ONLY complete Python code in a ```python block.
"""

META_ADVISOR_PROMPT = """You are a lessons-learned advisor for an evolutionary code optimization system.

## Your Role
Analyze FAILURES from recent evaluations. Your lessons get injected into LLM prompts to help future solutions avoid the same mistakes.

## What You're Given
- **Failure count**: How many candidates failed (crashes, invalid code, timeouts, etc.)
- **Error patterns**: Specific error messages encountered (including timeouts)
- **Previous lessons**: What you warned about last time

## Your Task: Write Concise Lessons (150-200 words max)

### Focus ONLY on Failure Prevention
You do NOT see successful solutions. Your job is purely defensive:
1. **Identify error patterns** - What mistakes are being made repeatedly?
2. **Explain root causes** - Why are these errors happening?
3. **Give specific fixes** - Exactly how to avoid each error type

### For Each Error Pattern:
- Quote the error briefly
- Explain what causes it
- Give a specific fix

## Output Format
Keep it SHORT and DIRECT:

**Avoid These Errors:**
- [Error pattern]: [How to fix]
- [Error pattern]: [How to fix]

---

{metrics_data}

Provide your lessons (150-200 words max):"""

# --- Test Inputs ---
INPUTS = generate_test_cases(num_tests=50)


# --- Score Function ---
def score_fn(compute_model_placement, inputs):
    """Evaluate placement algorithm: returns 0-100 score with sqrt scaling."""
    try:
        all_scores = []

        for gpu_num, models in inputs:
            # Run solution
            result = compute_model_placement(gpu_num, models)

            # Validate return type
            if not isinstance(result, dict):
                return {"error": f"Expected dict, got {type(result).__name__}"}

            # Validate all models placed exactly once
            placed = []
            for gpu_id, gpu_models in result.items():
                if not isinstance(gpu_models, list):
                    return {"error": f"GPU {gpu_id} value must be list, got {type(gpu_models).__name__}"}
                placed.extend(gpu_models)

            if len(placed) != len(models):
                return {"error": f"Not all models placed: {len(placed)}/{len(models)}"}

            # Validate model uniqueness (prevent reward hacking via duplicate models)
            placed_ids = [id(m) for m in placed]
            if len(set(placed_ids)) != len(placed_ids):
                return {"error": f"Duplicate models detected: {len(placed_ids) - len(set(placed_ids))} duplicates"}
            original_ids = {id(m) for m in models}
            if set(placed_ids) != original_ids:
                return {"error": "Placed models don't match input models (missing or foreign models)"}

            # Validate memory constraints
            for gpu_id, gpu_models in result.items():
                total_size = sum(m.model_size for m in gpu_models)
                if total_size > GPU_MEM_SIZE:
                    return {"error": f"GPU {gpu_id} exceeds memory: {total_size}GB > {GPU_MEM_SIZE}GB"}

            # Compute scores
            baseline_kvpr = calculate_kvpr(round_robin_placement(gpu_num, models))
            optimal_kvpr = compute_theoretical_optimal(gpu_num, models)
            solution_kvpr = calculate_kvpr(result)

            # Score with sqrt scaling
            if baseline_kvpr > optimal_kvpr:
                raw_ratio = (baseline_kvpr - solution_kvpr) / (baseline_kvpr - optimal_kvpr)
                test_score = 100.0 * (max(0.0, min(1.0, raw_ratio)) ** 0.5)
            else:
                test_score = 100.0 if solution_kvpr <= optimal_kvpr else 0.0

            all_scores.append(test_score)

        return {"score": sum(all_scores) / len(all_scores), "num_tests": len(all_scores)}

    except Exception as e:
        return {"error": str(e)}
