"""
Prompts for Transaction Scheduling evolution.
"""

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
def get_best_schedule(workload, num_seqs: int) -> tuple[int, list[int]]:
    '''
    Optimize schedule for a single workload.

    Args:
        workload: Workload object with num_txns, txns, get_opt_seq_cost
        num_seqs: Hint for number of sequences to try

    Returns:
        (makespan, schedule) where schedule is a permutation of [0, num_txns)
    '''
    pass
"""

SEED_PROGRAM = '''import random

def get_best_schedule(workload, num_seqs):
    """Find a good transaction schedule by trying random permutations."""
    n = workload.num_txns

    best_schedule = list(range(n))
    best_cost = workload.get_opt_seq_cost(best_schedule)

    for _ in range(num_seqs):
        schedule = list(range(n))
        random.shuffle(schedule)
        cost = workload.get_opt_seq_cost(schedule)
        if cost < best_cost:
            best_cost = cost
            best_schedule = schedule

    return best_cost, best_schedule
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
def get_best_schedule(workload, num_seqs: int) -> tuple[int, list[int]]:
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
