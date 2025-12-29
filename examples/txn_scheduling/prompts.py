"""
Prompts for Transaction Scheduling evolution.
"""

PROBLEM_DESCRIPTION = """
# Transaction Scheduling Optimization

## Problem
Optimize transaction scheduling for database workloads to minimize total makespan.

## Key Concepts
- Transactions have read (r) and write (w) operations on keys
- Example: "w-17 r-5 w-3" = write key 17, read key 5, write key 3
- Write-Write and Read-Write conflicts require waiting
- Read-Read can run in parallel (shared lock)

## Objective
Find the optimal ordering of 100 transactions to minimize makespan.
- Sequential baseline [0,1,...,99] gives ~452
- Better orderings reduce conflicts

## APIs
- `workload.num_txns` - Number of transactions (n)
- `workload.txns[i]` - Transaction i as list of (op_type, key, pos, txn_len)
- `workload.get_opt_seq_cost(seq)` - Compute makespan for ordering

## CRITICAL CONSTRAINTS
Do not sort transactions. Runtime MUST BE O(n) where n is the number of transactions.
Assume list operations are O(1), sequence copy operations are O(1), and get_opt_seq_cost is O(1).
Once a transaction is scheduled, it should NOT be moved. For example, local swap with transactions scheduled in a previous iteration is not allowed.
"""

FUNCTION_SIGNATURE = """
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload: Workload, num_seqs: int) -> tuple[int, list[int]]:
    '''Returns (makespan, schedule).'''
    pass

def get_random_costs() -> tuple[int, list[list[int]], list[int], float]:
    '''Returns (total_makespan, [sched1, sched2, sched3], [cost1, cost2, cost3], time).'''
    pass
"""

SEED_PROGRAM = '''# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3


def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using greedy cost sampling strategy.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    def get_greedy_cost_sampled(num_samples, sample_rate):
        # greedy with random starting point
        start_txn = random.randint(0, workload.num_txns - 1)
        txn_seq = [start_txn]
        remaining_txns = [x for x in range(0, workload.num_txns)]
        remaining_txns.remove(start_txn)
        running_cost = workload.txns[start_txn][0][3]
        for i in range(0, workload.num_txns - 1):
            min_cost = 100000 # MAX
            min_relative_cost = 10
            min_txn = -1
            holdout_txns = []
            done = False
            key_maps = []

            sample = random.random()
            if sample > sample_rate:
                idx = random.randint(0, len(remaining_txns) - 1)
                t = remaining_txns[idx]
                txn_seq.append(t)
                remaining_txns.pop(idx)
                continue

            for j in range(0, num_samples):
                idx = 0
                if len(remaining_txns) > 1:
                    idx = random.randint(0, len(remaining_txns) - 1)
                else:
                    done = True
                t = remaining_txns[idx]
                holdout_txns.append(remaining_txns.pop(idx))
                if workload.debug:
                    print(remaining_txns, holdout_txns)
                txn_len = workload.txns[t][0][3]
                test_seq = txn_seq.copy()
                test_seq.append(t)
                cost = 0
                cost = workload.get_opt_seq_cost(test_seq)
                if cost < min_cost:
                    min_cost = cost
                    min_txn = t
                if done:
                    break
            assert(min_txn != -1)
            running_cost = min_cost
            txn_seq.append(min_txn)
            holdout_txns.remove(min_txn)
            remaining_txns.extend(holdout_txns)

            if workload.debug:
                print("min: ", min_txn, remaining_txns, holdout_txns, txn_seq)
        if workload.debug:
            print(txn_seq)
            print(len(set(txn_seq)))
        assert len(set(txn_seq)) == workload.num_txns

        overall_cost = workload.get_opt_seq_cost(txn_seq)

        return overall_cost, txn_seq

    return get_greedy_cost_sampled(10, 1.0)

def get_random_costs():
    random.seed(42)  # Fixed seed for reproducibility
    start_time = time.time()
    workload_size = 100
    workload = Workload(WORKLOAD_1)

    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
    cost3 = workload3.get_opt_seq_cost(schedule3)
    print(cost1, cost2, cost3)
    return cost1 + cost2 + cost3, [schedule1, schedule2, schedule3], [cost1, cost2, cost3], time.time() - start_time


if __name__ == "__main__":
    makespan, schedule, costs, elapsed = get_random_costs()
    print(f"Makespan: {makespan}, Costs: {costs}, Time: {elapsed}")
'''

# No inspiration seeds - matching OpenEvolve which only has a single seed program
SEED_INSPIRATIONS = []

DIVERSITY_SEED_PROMPT = """
# Transaction Scheduling Optimization

## Problem
Optimize transaction scheduling for database workloads to minimize total makespan.

## Key Concepts
- Transactions have read (r) and write (w) operations on keys
- Write-Write and Read-Write conflicts require waiting
- Read-Read can run in parallel (shared lock)

## Objective
Find the optimal ordering of 100 transactions to minimize makespan.

## CRITICAL CONSTRAINTS
- Runtime MUST be O(n) where n is the number of transactions
- Do not sort transactions
- Once a transaction is scheduled, it should NOT be moved

## APIs
- `workload.num_txns` - Number of transactions (n)
- `workload.txns[i]` - Transaction i as list of (op_type, key, pos, txn_len)
- `workload.get_opt_seq_cost(seq)` - Compute makespan for ordering

## Function Signature
```python
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload: Workload, num_seqs: int) -> tuple[int, list[int]]:
    '''Returns (makespan, schedule).'''
    pass

def get_random_costs() -> tuple[int, list[list[int]], list[int], float]:
    '''Returns (total_makespan, [sched1, sched2, sched3], [cost1, cost2, cost3], time).'''
    pass
```

## Your Task
Generate a solution with DIFFERENT BEHAVIORAL CHARACTERISTICS than the existing seeds.

**CRITICAL: BEHAVIORAL DIVERSITY IS ESSENTIAL.**

Focus on creating solutions that exhibit different runtime behaviors:
- Different execution time profiles (fast approximate vs slower precise)
- Different memory usage patterns
- Different tradeoffs between exploration and exploitation
- Different sensitivity to workload characteristics

The goal is behavioral variety in the population, not just different code.

## Existing Seeds (aim for different behavioral characteristics):
{existing_seeds}

## Instructions
1. Review the existing seeds and consider their likely runtime behavior
2. Design a solution that would behave differently (e.g., different speed/quality tradeoff)
3. Implement it as a complete, working solution
4. Output ONLY the complete Python code in a ```python block
"""

META_ADVISOR_PROMPT = """You are a meta-advisor for an evolutionary code optimization system.

## Your Role
Analyze evolution metrics and provide strategic guidance. Your advice gets injected into LLM prompts to steer the next generation of solutions.

## What You're Given
- **Period Metrics**: Acceptance/rejection/error rates from recent evaluations
- **Error Messages**: Specific failure patterns to avoid
- **Previous Advice**: What you recommended last time
- **Best Solution**: Current top performer's code
- **Progress**: Budget consumption percentage

## Your Task: Write Strategic Advice (400-500 words)

### 1. Reflect on Previous Advice
- Look at the metrics: did your last advice help or hurt?
- If acceptance rate improved → reinforce what worked
- If errors increased → explicitly retract problematic suggestions
- If no change → your advice may have been too vague, be more specific

### 2. Interpret the Metrics
Diagnose what the numbers tell you:
- **High rejection, low error**: Valid code but not improving → need MORE diversity, structural changes
- **High error rate**: Bugs in generated code → identify patterns from error messages, warn against them
- **Low acceptance + plateau**: Archive saturated → need fundamentally different algorithmic approaches
- **Good acceptance rate**: Current direction working → encourage deeper exploration of similar ideas

### 3. Analyze the Best Solution
Look at the provided code:
- What is its core algorithmic strategy?
- What are its likely weaknesses or blind spots?
- What aspects of the problem might it be ignoring?
- Suggest exploring what it DOESN'T do

### 4. Give Actionable Direction
Be SPECIFIC about what to try differently. Bad advice: "try something different." Good advice:
- "The best solution builds schedules incrementally. Try approaches that evaluate complete schedules first, then refine."
- "Current solutions focus on [X]. Consider approaches that optimize for [Y] instead."
- "Error patterns show [issue]. Ensure your solution handles [specific case]."

### 5. Warn Against Failure Patterns
Based on error messages, give explicit warnings:
- Quote specific error patterns and explain how to avoid them
- If timeout errors: suggest ways to reduce computational complexity
- If assertion errors: highlight what invariants must be maintained

## Critical Rules
- Each LLM sees a DIFFERENT parent solution (not the best one)
- Tell them to STRUCTURALLY modify their given parent
- Don't tell them to copy the best solution
- Push for paradigm changes, not parameter tweaks
- Your advice should evolve based on what's working/failing

## Output Format
Structure your advice clearly:
1. **What's Working** (based on metrics)
2. **What to Avoid** (based on errors)
3. **Strategic Direction** (what to try next)
4. **Specific Suggestions** (2-3 concrete ideas derived from analysis)

---

{metrics_data}

Provide your strategic advice:"""
