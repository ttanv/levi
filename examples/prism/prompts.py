"""
Prompts for PRISM (ML Model Placement) evolution.
"""

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
