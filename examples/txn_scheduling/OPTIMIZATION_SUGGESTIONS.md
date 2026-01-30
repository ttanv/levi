# Multi-Island PE Optimization Suggestions

## Current Performance Summary

| Run | Budget | Best Score | Islands | PE Events | Gap (max-min) |
|-----|--------|-----------|---------|-----------|---------------|
| 20260126_032137 | $4.00 | **69.93** | 3 | 138 | 0.45 |
| 20260125_203335 | $6.90 | 69.47 | 3 | 254 | 0.91 |
| 20260126_084138 | $3.89 | 69.47 | 3 | 92 | 0.45 |
| 20260126_131418 | $3.90 | 68.11 | 10 | 28 | 2.94 |
| 20260126_062653 | $0.78 | 66.76 | 3 | 21 | 5.88 |

**Key observation**: Best scores (69.47-69.93) achieved with 3 islands, $4 budget, tight convergence (gap < 1.0).

---

## Critical Bug: Prompt-Acceptance Mismatch

**Location**: `algoforge/island/multi_island_pe.py` lines 466, 484, 558

**The Problem**:
```python
# Line 466: Gets STRONGEST island score
best_score = sorted_islands[-1][1]  # e.g., 69.5

# Line 484: Tells LLM to beat STRONGEST
prompt = f"Generate a solution that scores HIGHER than {best_score:.1f}."

# Line 558: But accepts if beats WEAKEST
if paradigm_score > weakest_score:  # e.g., > 68.6
```

**Impact**: When islands converge (gap = 0.45), we ask LLM to beat 69.5 but only need 69.05. This makes the prompt ~0.45 points harder than necessary - significant when improvements are incremental.

**Fix**:
```python
# Option A: Change prompt to match acceptance (use weakest)
weakest_score = sorted_islands[0][1]
prompt = f"Generate a solution that scores HIGHER than {weakest_score:.1f}."

# Option B: Change acceptance to match prompt (use strongest)
# This is stricter but more honest
if paradigm_score > best_score:
```

**Recommendation**: Use Option A - it's honest about the actual bar to clear.

---

## Why Paradigm Shift Becomes Useless Late in Runs

### Problem 1: Diminishing Returns from Convergence
As islands converge (gap drops from 17 to 0.45), the acceptance threshold rises:
- Early: Beat 48.0 to enter weakest island (easy)
- Late: Beat 69.0 to enter weakest island (nearly impossible)

### Problem 2: Algorithmic Plateau
All high-scoring solutions (68+) use the **same paradigm**:
1. Conflict matrix construction
2. Greedy initialization with priority
3. Local search (swap, 2-opt, insertion)
4. Time-limited iteration

There's no "fundamentally different approach" left to discover - the solution space is exhausted.

### Problem 3: Prompt Doesn't Guide Refinement
Late-stage prompt says "Focus on REFINEMENT" but provides no specific guidance on *what* to refine. LLM generates yet another conflict-greedy-localsearch variant.

---

## Specific Fixes (Budget-Conscious)

### 1. Fix the Prompt-Acceptance Bug (Zero Cost)
```python
# In trigger_cross_island_pe(), change line 466:
weakest_score = sorted_islands[0][1]  # Use weakest, not strongest
# And update line 484 to use weakest_score instead of best_score
```

### 2. Adaptive PE Frequency (Zero Cost)
Reduce PE frequency as islands converge - saves budget for mutations:
```python
# In _eval_consumer(), replace fixed pe_interval check:
island_gap = max(island_scores) - min(island_scores)
effective_interval = self.pe_interval if island_gap > 2.0 else self.pe_interval * 3
if self.state.eval_count % effective_interval == 0:
    await self.trigger_cross_island_pe(executor)
```

### 3. Late-Stage PE Prompt Improvement (Zero Cost)
Replace generic "Focus on REFINEMENT" with specific guidance:
```python
late_stage_guidance = """
Focus on REFINEMENT of the best approach:
- Tune numeric constants (cooling rate, beam width, iteration limits)
- Add caching for repeated conflict calculations
- Improve initialization (try multiple random starts)
- Strengthen local search (larger neighborhood, more iterations)
- Add early termination when no improvement for N iterations

DO NOT try fundamentally different algorithms at this stage.
"""
```

### 4. Disable PE After 70% Budget (Saves ~15% Budget)
PE becomes wasteful when islands have converged:
```python
if budget_pct >= 70 and island_gap < 1.0:
    logger.info("[PE] Skipping PE - islands converged, focusing on mutations")
    return stats
```

### 5. Culling Strategy Adjustment (Zero Cost)
Current: Cull at 25%, 50%, 75% milestones.
Problem: With 3 islands, culling removes 33% diversity each time.

**Alternative**: Only cull if weakest is >5 points behind:
```python
if weakest_score < strongest_score - 5.0:
    self.cull_weakest_island()
```

---

## Problem-Specific Optimizations

### Transaction Scheduling Domain Knowledge

The problem has specific structure that prompts should exploit:

1. **Conflict Graph Insight**: Transactions form a conflict graph. Optimal scheduling is related to graph coloring / topological sort.

2. **Key Access Patterns**: Transactions that access disjoint key sets can run in parallel. Clustering by key access is powerful.

3. **Critical Path**: The longest chain of conflicting transactions determines minimum makespan.

### Enhanced PE Prompt for This Problem
```python
DOMAIN_SPECIFIC_PE_PROMPT = """
## Domain-Specific Hints for Transaction Scheduling

The problem has exploitable structure:

1. **Conflict Graph**: Build adjacency matrix of W-W and R-W conflicts.
   - Transactions with no conflicts can be scheduled freely
   - High-conflict transactions should be spread apart

2. **Key Clustering**: Group transactions by key access sets.
   - Non-overlapping clusters can interleave optimally
   - Use set intersection to measure overlap

3. **Critical Path**: Find longest conflict chain.
   - This is the theoretical minimum makespan
   - Schedule critical path first, fill gaps with others

4. **Proven Techniques** (from best solutions):
   - Weighted conflict scoring: W-W conflicts = 100, R-W = 50
   - Beam search with width 50-100
   - Multi-start greedy (5-10 random initializations)
   - Simulated annealing: alpha=0.998, slow cooling
   - Local search moves: swap, insert, 2-opt

Your solution MUST include:
- Conflict analysis in O(n^2) or better
- At least 2 local search move types
- Time-limited iteration with early termination
"""
```

### Enhanced Mutation Prompts
```json
{
  "mutation": {
    "mimo": "Improve the transaction scheduling algorithm.\n\nKey optimizations to consider:\n1. Conflict-aware initialization (schedule low-conflict txns first)\n2. Stronger local search (try swap + insert + 2-opt)\n3. Better termination (stop when no improvement for 50 iterations)\n4. Caching (precompute conflict matrix once)\n\nOutput complete improved code.",

    "deepseek": "You are optimizing a transaction scheduling algorithm.\n\n**Current approach analysis:**\n- Identify the initialization strategy\n- Identify the local search moves used\n- Identify the termination condition\n\n**Improvement targets:**\n- If initialization is random, make it conflict-aware\n- If only using swaps, add insertion moves\n- If termination is time-based, add no-improvement early exit\n- If conflicts are recomputed each time, add caching\n\n**Critical**: The score function calls get_opt_seq_cost() which is expensive. Minimize calls by:\n- Caching partial results\n- Using delta evaluation for moves\n- Early termination when stuck\n\nOutput complete improved Python code.",

    "qwen": "Improve the scheduling algorithm.\n\nFocus on:\n1. Smarter conflict handling (W-W=100 weight, R-W=50 weight)\n2. Multi-start strategy (try 5 different random seeds)\n3. Hybrid local search (swap + insert moves)\n4. Adaptive iteration (more iterations when improving)\n\nOutput complete code."
  }
}
```

---

## Experimental Ideas (Higher Risk/Reward)

### 1. Elite Injection Instead of PE
Instead of generating new solutions, inject the global best into lagging islands:
```python
# Every 50 evals, copy best elite to all islands
if self.state.eval_count % 50 == 0:
    best_code, best_score = self.get_global_best()
    for island in self.islands:
        if island._best_score < best_score - 2.0:
            island.inject_elite(best_code, best_score)
```

### 2. Hyperparameter Mutation
Late-stage: Extract numeric constants from best solution and perturb them:
```python
# Find patterns like "alpha = 0.998" and generate variants
# alpha = 0.995, 0.997, 0.999, etc.
```

### 3. Component Recombination
Parse best solutions into components (init, search, termination) and recombine:
```python
# Solution A: greedy_init + swap_search + time_termination
# Solution B: random_init + beam_search + convergence_termination
# Generate: greedy_init + beam_search + convergence_termination
```

### 4. Self-Improving Prompt
Track which prompt modifications led to accepted solutions and reinforce them.

---

## Recommended Next Run Configuration

```python
# In run_multi_island_pe.py

BUDGET = 4.00  # Sweet spot from analysis

# Keep 3 islands - proven optimal
n_islands = 3

# Reduce PE interval for more events early, but...
pe_interval = 10

# Add to config: adaptive PE and late-stage skip
# (requires code changes to multi_island_pe.py)
```

### Code Changes Checklist

1. [x] ~~Fix prompt-acceptance mismatch~~ - Replaced with optimized prompt (no explicit score target)
2. [ ] Add adaptive PE frequency based on island gap
3. [x] Skip PE after 60% budget (implemented)
4. [x] Replaced PE prompt with optimized version from optimized_prompts.json
5. [ ] Update optimized_prompts.json with domain-specific hints
6. [ ] Add PE statistics logging (accepted/rejected/scores) to snapshot

### Additional Changes Made (Jan 26)

- **Islands**: 3 (was 10)
- **Workers**: 12 LLM + 12 eval (was 8 each)
- **Gemini Flash 3**: Added as mutator (temp=0.3, reasoning off)
- **Culling milestones**: 50%, 75%, 88% (was 25%, 50%, 75%)
- **PE cutoff**: Stops at 60% budget

---

## Expected Impact

| Change | Effort | Expected Gain |
|--------|--------|---------------|
| Fix prompt-acceptance bug | 5 min | +0.5-1.0 points |
| Adaptive PE frequency | 15 min | Save 10-15% budget |
| Skip late PE | 5 min | Save 15-20% budget |
| Domain-specific prompts | 30 min | +0.5-1.5 points |
| Better mutation prompts | 20 min | +0.3-0.5 points |

**Total potential**: 70.5-72.0 score with same budget, or 69.5+ with lower budget.

---

## Score Ceiling Analysis

Based on the scoring formula in `problem.py`:
```python
BASELINE = 452 (sequential)
EFFECTIVE_OPTIMAL = OPTIMAL + 0.10 * (BASELINE - OPTIMAL)
score = ((BASELINE - total) / (BASELINE - EFFECTIVE_OPTIMAL)) * 100
```

Current best makespan at 69.93 score implies total makespan ~180-190.
Theoretical optimal (all max-length txns in parallel) would be ~30-50.

The 69-70 plateau suggests we're hitting algorithmic limits, not prompt limits. To break through:
1. Need fundamentally better algorithms (ILP, constraint programming)
2. Or problem-specific insights we haven't discovered yet

The remaining 30 points likely require exponential search that isn't practical within budget.
