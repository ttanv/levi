"""
Transaction Scheduling Problem Definition.

Contains problem description, prompts, scoring function, and test inputs.
"""

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# --- Prompts ---

PROBLEM_DESCRIPTION = """
# Transaction Scheduling Optimization

## Problem
Optimize transaction scheduling for database workloads to minimize total makespan.

## Key Concepts
- Transactions have read (r) and write (w) operations on keys
- Write-Write and Read-Write conflicts require waiting
- Read-Read can run in parallel (shared lock)

## Objective
Find the optimal ordering of 100 transactions to minimize makespan.
- Sequential baseline [0,1,...,99] gives ~452
- Better orderings reduce conflicts

## Input
A workload object with:
- `workload.num_txns` - Number of transactions (n)
- `workload.txns[i]` - Transaction i as list of (op_type, key, pos, txn_len)
- `workload.get_opt_seq_cost(seq)` - Compute makespan for ordering

## Evaluation
Your function is called on 3 workloads. Total makespan is summed.

## You can import standard library modules (random, heapq, collections, math, numpy, etc.)
"""

FUNCTION_SIGNATURE = """
def get_best_schedule(workload) -> tuple[int, list[int]]:
    '''
    Optimize schedule for a single workload.

    Args:
        workload: Workload object with num_txns, txns, get_opt_seq_cost

    Returns:
        (makespan, schedule) where schedule is a permutation of [0, num_txns)
    '''
    pass
"""

SEED_PROGRAM = '''import random

def get_best_schedule(workload):
    """Get optimal schedule using greedy cost sampling strategy."""
    random.seed(42)

    start_txn = random.randint(0, workload.num_txns - 1)
    txn_seq = [start_txn]
    remaining_txns = list(range(workload.num_txns))
    remaining_txns.remove(start_txn)

    for _ in range(workload.num_txns - 1):
        min_cost = float('inf')
        min_txn = -1
        holdout_txns = []

        num_samples = min(10, len(remaining_txns))
        for _ in range(num_samples):
            if not remaining_txns:
                break
            idx = random.randint(0, len(remaining_txns) - 1)
            t = remaining_txns[idx]
            holdout_txns.append(remaining_txns.pop(idx))

            test_seq = txn_seq + [t]
            cost = workload.get_opt_seq_cost(test_seq)
            if cost < min_cost:
                min_cost = cost
                min_txn = t

        if min_txn != -1:
            txn_seq.append(min_txn)
            holdout_txns.remove(min_txn)
        remaining_txns.extend(holdout_txns)

    return workload.get_opt_seq_cost(txn_seq), txn_seq
'''

SEED_INSPIRATIONS = []

DIVERSITY_SEED_PROMPT = """
# Transaction Scheduling Optimization

## Problem
Optimize transaction scheduling for database workloads to minimize total makespan.

## Key Concepts
- Transactions have read (r) and write (w) operations on keys
- Write-Write and Read-Write conflicts require waiting
- Read-Read can run in parallel (shared lock)

## Input
A workload object with:
- `workload.num_txns` - Number of transactions (n)
- `workload.txns[i]` - Transaction i as list of (op_type, key, pos, txn_len)
- `workload.get_opt_seq_cost(seq)` - Compute makespan for ordering

## Function Signature
```python
def get_best_schedule(workload) -> tuple[int, list[int]]:
    '''Returns (makespan, schedule) for a single workload.'''
    pass
```

## You can import standard library modules (random, heapq, collections, math, numpy, etc.)

## Your Task: ALGORITHMIC DIVERSITY

Design a solution using a **FUNDAMENTALLY DIFFERENT ALGORITHM** than the existing seeds.

**DO NOT:**
- Make minor variations or parameter tweaks
- Use the same core algorithm with different constants

**DO:**
- Analyze what paradigm each existing seed uses
- Design from first principles using a different strategy
- Consider what information the existing seeds are NOT using

## Existing Seeds:
{existing_seeds}

## Output
Output ONLY the complete Python code in a ```python block.
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

INPUTS = [Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)]

# Scoring reference points: sequential ordering = 0 pts, theoretical optimal = 100 pts
BASELINE = sum(w.get_opt_seq_cost(list(range(w.num_txns))) for w in INPUTS)
OPTIMAL = sum(max(txn[0][3] for txn in w.txns) for w in INPUTS)
EFFECTIVE_OPTIMAL = OPTIMAL + 0.10 * (BASELINE - OPTIMAL)  # Shifted to make 100 achievable


# --- Score Function ---

def score_fn(get_best_schedule, inputs):
    """Evaluate scheduling algorithm: returns 0-100 score based on total makespan."""
    try:
        total = 0
        for w in inputs:
            _, schedule = get_best_schedule(w)
            if set(schedule) != set(range(w.num_txns)):
                return {"error": "Invalid schedule: not a permutation"}
            total += w.get_opt_seq_cost(schedule)

        if total >= BASELINE:
            score = 0.0
        elif total <= EFFECTIVE_OPTIMAL:
            score = 100.0
        else:
            score = ((BASELINE - total) / (BASELINE - EFFECTIVE_OPTIMAL)) * 100

        return {"score": score, "makespan": total}
    except Exception as e:
        return {"error": str(e)}
