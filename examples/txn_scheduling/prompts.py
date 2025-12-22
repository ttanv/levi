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
- `workload.num_txns` - Number of transactions
- `workload.txns[i]` - Transaction i as list of (op_type, key, pos, txn_len)
- `workload.get_opt_seq_cost(seq)` - Compute makespan for ordering

## CRITICAL: Execution Speed
Your code MUST run fast. Solutions that timeout are rejected.
- Avoid calling get_opt_seq_cost() in tight loops (it's expensive)
- Prefer lower time complexity algorithms
- Cache results, precompute data, use efficient data structures

## Strategies to Try
- Conflict graph analysis
- Greedy heuristics (min incremental cost)
- Local search (swaps, insertions)
- Simulated annealing
- Sorting by transaction properties (length, conflicts)
"""

FUNCTION_SIGNATURE = """
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload: Workload, num_seqs: int) -> tuple[int, list[int]]:
    '''Returns (makespan, schedule).'''
    pass

def get_random_costs() -> tuple[int, list[list[int]], float]:
    '''Returns (total_makespan, [sched1, sched2, sched3], time).'''
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
    return cost1 + cost2 + cost3, [schedule1, schedule2, schedule3], time.time() - start_time


if __name__ == "__main__":
    makespan, schedule, time = get_random_costs()
    print(f"Makespan: {makespan}, Time: {time}")
'''

# Inspiration seeds varying in code_length, loop_count, cyclomatic_complexity
SEED_INSPIRATIONS = [
    # === SIMPLE SEEDS (5) ===
    # 1. Minimal: just return sequential order (very short, no loops, no branches)
    '''import time
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload, num_seqs):
    seq = list(range(workload.num_txns))
    return workload.get_opt_seq_cost(seq), seq

def get_random_costs():
    start = time.time()
    w1, w2, w3 = Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)
    c1, s1 = get_best_schedule(w1, 1)
    c2, s2 = get_best_schedule(w2, 1)
    c3, s3 = get_best_schedule(w3, 1)
    return c1 + c2 + c3, [s1, s2, s3], time.time() - start
''',

    # 2. Random shuffle: one loop, minimal branching
    '''import time
import random
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload, num_seqs):
    best_cost, best_seq = float('inf'), None
    for _ in range(num_seqs):
        seq = list(range(workload.num_txns))
        random.shuffle(seq)
        cost = workload.get_opt_seq_cost(seq)
        if cost < best_cost:
            best_cost, best_seq = cost, seq
    return best_cost, best_seq

def get_random_costs():
    random.seed(42)
    start = time.time()
    w1, w2, w3 = Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)
    c1, s1 = get_best_schedule(w1, 20)
    c2, s2 = get_best_schedule(w2, 20)
    c3, s3 = get_best_schedule(w3, 20)
    return c1 + c2 + c3, [s1, s2, s3], time.time() - start
''',

    # 3. Sort by txn length: one loop, one sort
    '''import time
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload, num_seqs):
    seq = sorted(range(workload.num_txns), key=lambda i: workload.txns[i][0][3])
    return workload.get_opt_seq_cost(seq), seq

def get_random_costs():
    start = time.time()
    w1, w2, w3 = Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)
    c1, s1 = get_best_schedule(w1, 1)
    c2, s2 = get_best_schedule(w2, 1)
    c3, s3 = get_best_schedule(w3, 1)
    return c1 + c2 + c3, [s1, s2, s3], time.time() - start
''',

    # 4. Reverse order: trivially different from sequential
    '''import time
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload, num_seqs):
    seq = list(range(workload.num_txns - 1, -1, -1))
    return workload.get_opt_seq_cost(seq), seq

def get_random_costs():
    start = time.time()
    w1, w2, w3 = Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)
    c1, s1 = get_best_schedule(w1, 1)
    c2, s2 = get_best_schedule(w2, 1)
    c3, s3 = get_best_schedule(w3, 1)
    return c1 + c2 + c3, [s1, s2, s3], time.time() - start
''',

    # 5. Adjacent swap hill climb: nested loops, more complexity
    '''import time
import random
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload, num_seqs):
    seq = list(range(workload.num_txns))
    random.shuffle(seq)
    cost = workload.get_opt_seq_cost(seq)
    for _ in range(50):
        for i in range(len(seq) - 1):
            seq[i], seq[i+1] = seq[i+1], seq[i]
            new_cost = workload.get_opt_seq_cost(seq)
            if new_cost < cost:
                cost = new_cost
            else:
                seq[i], seq[i+1] = seq[i+1], seq[i]
    return cost, seq

def get_random_costs():
    random.seed(42)
    start = time.time()
    w1, w2, w3 = Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)
    c1, s1 = get_best_schedule(w1, 1)
    c2, s2 = get_best_schedule(w2, 1)
    c3, s3 = get_best_schedule(w3, 1)
    return c1 + c2 + c3, [s1, s2, s3], time.time() - start
''',

    # === COMPLEX SEEDS (5) ===
    # 6. Greedy insertion with conflict analysis
    '''import time
import random
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload, num_seqs):
    """Greedy insertion: build schedule by adding txn with minimal cost increase."""
    n = workload.num_txns
    seq = []
    remaining = set(range(n))

    # Start with random txn
    start_txn = random.choice(list(remaining))
    seq.append(start_txn)
    remaining.remove(start_txn)

    # Greedily add remaining txns
    while remaining:
        best_txn = None
        best_cost = float('inf')
        current_cost = workload.get_opt_seq_cost(seq)

        # Try each remaining txn at end of current sequence
        for txn in remaining:
            test_seq = seq + [txn]
            new_cost = workload.get_opt_seq_cost(test_seq)
            delta = new_cost - current_cost

            if delta < best_cost:
                best_cost = delta
                best_txn = txn

        seq.append(best_txn)
        remaining.remove(best_txn)

    final_cost = workload.get_opt_seq_cost(seq)
    return final_cost, seq

def get_random_costs():
    random.seed(42)
    start = time.time()
    w1, w2, w3 = Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)
    c1, s1 = get_best_schedule(w1, 1)
    c2, s2 = get_best_schedule(w2, 1)
    c3, s3 = get_best_schedule(w3, 1)
    return c1 + c2 + c3, [s1, s2, s3], time.time() - start
''',

    # 7. Multi-start greedy with local improvement
    '''import time
import random
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload, num_seqs):
    """Multi-start greedy: try multiple random starts, keep best."""
    n = workload.num_txns
    best_overall_cost = float('inf')
    best_overall_seq = None

    for trial in range(max(3, num_seqs // 5)):
        # Random greedy construction
        seq = []
        remaining = list(range(n))
        random.shuffle(remaining)

        for _ in range(n):
            if len(remaining) == 1:
                seq.append(remaining[0])
                break

            # Sample a few candidates
            sample_size = min(8, len(remaining))
            candidates = random.sample(remaining, sample_size)

            best_txn = candidates[0]
            best_cost = float('inf')

            for txn in candidates:
                test_seq = seq + [txn]
                cost = workload.get_opt_seq_cost(test_seq)
                if cost < best_cost:
                    best_cost = cost
                    best_txn = txn

            seq.append(best_txn)
            remaining.remove(best_txn)

        # Local improvement with swaps
        improved = True
        iterations = 0
        while improved and iterations < 20:
            improved = False
            iterations += 1
            current_cost = workload.get_opt_seq_cost(seq)

            for i in range(n - 1):
                seq[i], seq[i+1] = seq[i+1], seq[i]
                new_cost = workload.get_opt_seq_cost(seq)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    seq[i], seq[i+1] = seq[i+1], seq[i]

        trial_cost = workload.get_opt_seq_cost(seq)
        if trial_cost < best_overall_cost:
            best_overall_cost = trial_cost
            best_overall_seq = seq[:]

    return best_overall_cost, best_overall_seq

def get_random_costs():
    random.seed(42)
    start = time.time()
    w1, w2, w3 = Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)
    c1, s1 = get_best_schedule(w1, 10)
    c2, s2 = get_best_schedule(w2, 10)
    c3, s3 = get_best_schedule(w3, 10)
    return c1 + c2 + c3, [s1, s2, s3], time.time() - start
''',

    # 8. Simulated annealing scheduler
    '''import time
import random
import math
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload, num_seqs):
    """Simulated annealing: escape local optima with probabilistic acceptance."""
    n = workload.num_txns

    # Initial solution
    current_seq = list(range(n))
    random.shuffle(current_seq)
    current_cost = workload.get_opt_seq_cost(current_seq)

    best_seq = current_seq[:]
    best_cost = current_cost

    # Annealing schedule
    temp = 50.0
    cooling_rate = 0.95
    iterations_per_temp = max(20, n)

    while temp > 0.1:
        for _ in range(iterations_per_temp):
            # Generate neighbor: swap two random positions
            i, j = random.sample(range(n), 2)
            current_seq[i], current_seq[j] = current_seq[j], current_seq[i]

            new_cost = workload.get_opt_seq_cost(current_seq)
            delta = new_cost - current_cost

            # Accept if better, or with probability based on temp
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_seq = current_seq[:]
            else:
                # Revert swap
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]

        temp *= cooling_rate

    return best_cost, best_seq

def get_random_costs():
    random.seed(42)
    start = time.time()
    w1, w2, w3 = Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)
    c1, s1 = get_best_schedule(w1, 1)
    c2, s2 = get_best_schedule(w2, 1)
    c3, s3 = get_best_schedule(w3, 1)
    return c1 + c2 + c3, [s1, s2, s3], time.time() - start
''',

    # 9. Beam search with position exploration
    '''import time
import random
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload, num_seqs):
    """Beam search: maintain top-k partial schedules, expand best ones."""
    n = workload.num_txns
    beam_width = 4

    # Initialize beam with random starting transactions
    beam = []
    starts = random.sample(range(n), min(beam_width, n))
    for start_txn in starts:
        seq = [start_txn]
        cost = workload.get_opt_seq_cost(seq)
        beam.append((cost, seq, set([start_txn])))

    # Expand beam until all schedules are complete
    for step in range(n - 1):
        candidates = []

        for current_cost, current_seq, used in beam:
            remaining = set(range(n)) - used

            # Try adding each remaining transaction
            sample_size = min(10, len(remaining))
            for txn in random.sample(list(remaining), sample_size):
                new_seq = current_seq + [txn]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_used = used | {txn}
                candidates.append((new_cost, new_seq, new_used))

        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    # Return best complete schedule
    best_cost, best_seq, _ = min(beam, key=lambda x: x[0])
    return best_cost, best_seq

def get_random_costs():
    random.seed(42)
    start = time.time()
    w1, w2, w3 = Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)
    c1, s1 = get_best_schedule(w1, 1)
    c2, s2 = get_best_schedule(w2, 1)
    c3, s3 = get_best_schedule(w3, 1)
    return c1 + c2 + c3, [s1, s2, s3], time.time() - start
''',

    # 10. Conflict-aware priority scheduling
    '''import time
import random
from collections import defaultdict
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload, num_seqs):
    """Priority-based: analyze conflicts, schedule low-conflict txns first."""
    n = workload.num_txns

    # Build conflict graph: count conflicts between transaction pairs
    conflict_count = defaultdict(int)
    for i in range(n):
        txn_i = workload.txns[i]
        keys_i = set()
        writes_i = set()

        for op_type, key, _, _ in txn_i:
            keys_i.add(key)
            if op_type == 'w':
                writes_i.add(key)

        for j in range(i + 1, n):
            txn_j = workload.txns[j]
            for op_type, key, _, _ in txn_j:
                # Write-write or write-read conflict
                if key in writes_i or (op_type == 'w' and key in keys_i):
                    conflict_count[i] += 1
                    conflict_count[j] += 1

    # Sort by conflict count (ascending) with randomization for ties
    txns_by_conflicts = sorted(range(n), key=lambda t: (conflict_count[t], random.random()))

    # Try multiple orderings based on conflict priority
    best_cost = float('inf')
    best_seq = txns_by_conflicts

    for trial in range(max(2, num_seqs // 10)):
        # Partition into low/medium/high conflict
        third = n // 3
        low_conflict = txns_by_conflicts[:third]
        med_conflict = txns_by_conflicts[third:2*third]
        high_conflict = txns_by_conflicts[2*third:]

        # Shuffle within each partition
        random.shuffle(low_conflict)
        random.shuffle(med_conflict)
        random.shuffle(high_conflict)

        # Try different orderings
        for order in [
            low_conflict + med_conflict + high_conflict,
            low_conflict + high_conflict + med_conflict,
            high_conflict + low_conflict + med_conflict,
        ]:
            cost = workload.get_opt_seq_cost(order)
            if cost < best_cost:
                best_cost = cost
                best_seq = order

    return best_cost, best_seq

def get_random_costs():
    random.seed(42)
    start = time.time()
    w1, w2, w3 = Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)
    c1, s1 = get_best_schedule(w1, 10)
    c2, s2 = get_best_schedule(w2, 10)
    c3, s3 = get_best_schedule(w3, 10)
    return c1 + c2 + c3, [s1, s2, s3], time.time() - start
''',
]
