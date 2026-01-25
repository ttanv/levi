---
name: AlgoForge Diversity Analysis
description: This skill should be used when the user asks to "analyze diversity", "check algorithmic diversity", "analyze snapshot", "examine behavioral diversity", "review paradigm families", "analyze algoforge run", "check elite population", or wants to understand the variety of algorithmic approaches in an AlgoForge CVT-MAP-Elites archive.
version: 1.0.0
---

# AlgoForge Diversity Analysis

This skill provides comprehensive analysis of algorithmic and behavioral diversity in AlgoForge CVT-MAP-Elites archives.

## Overview

AlgoForge uses CVT-MAP-Elites (Centroidal Voronoi Tessellation + Quality-Diversity) to evolve diverse, high-performing algorithms. This skill analyzes `snapshot.json` files from runs to assess:

1. **Behavioral Diversity** - How well algorithms spread across the behavior space
2. **Algorithmic Diversity** - Whether solutions represent genuinely different paradigms
3. **Score Distribution** - Quality profiles across the archive
4. **Paradigm Classification** - Grouping solutions by algorithmic approach

## When to Use This Skill

- After an AlgoForge run completes, to evaluate archive diversity
- When comparing runs with different configurations
- To identify if Punctuated Equilibrium is producing genuine paradigm shifts
- To understand why certain behavioral regions have no elites
- To verify behavioral noise settings aren't artificially inflating diversity

## Analysis Procedure

### Step 1: Load Snapshot Data

Read the snapshot.json file from the run directory:

```python
import json
with open('runs/YYYYMMDD_HHMMSS/snapshot.json') as f:
    snapshot = json.load(f)
```

Key fields:
- `elites`: List of elite programs with scores, code, and behavior vectors
- `cells`: CVT cell information
- `stats`: Run statistics

### Step 2: Behavioral Diversity Metrics

Calculate diversity metrics from behavior vectors:

```python
elites = snapshot['elites']
n_elites = len(elites)
n_unique_cells = len(set(e['cell_id'] for e in elites))

# Extract behavior vectors
behavior_vectors = [list(e['behavior'].values()) for e in elites]
```

Key metrics:
- **Elite Count**: Total surviving elites
- **Unique Cells**: Number of distinct behavioral niches occupied
- **Coverage**: `n_unique_cells / n_centroids` - fraction of behavior space covered

### Step 3: Paradigm Classification

Classify algorithms into paradigm families by analyzing code structure:

**Major Paradigm Indicators:**

| Paradigm | Code Markers |
|----------|--------------|
| Binary Search | `while lo < hi`, `mid = (lo + hi)`, bisect logic |
| Simulated Annealing | `temperature`, `exp(-delta/T)`, `random() < prob` |
| Genetic Algorithm | `population`, `crossover`, `mutation`, `fitness` |
| Tabu Search | `tabu_list`, `tabu_tenure`, recent move tracking |
| Greedy | Single-pass assignment without backtracking |
| Dynamic Programming | `memo`, `dp[]`, subproblem caching |
| Local Search | `improve`, `neighbor`, hill climbing |
| Bin Packing variants | FFD, BFD, WFD patterns |

**Classification Algorithm:**

```python
def classify_paradigm(code):
    code_lower = code.lower()

    # Check for binary search indicators
    if 'while lo' in code_lower or 'while left' in code_lower:
        if 'mid' in code_lower:
            return 'Binary Search'

    # Check for simulated annealing
    if 'temperature' in code_lower and 'exp(' in code_lower:
        return 'Simulated Annealing'

    # Check for genetic algorithm
    if 'population' in code_lower and 'crossover' in code_lower:
        return 'Genetic Algorithm'

    # Check for tabu search
    if 'tabu' in code_lower:
        return 'Tabu Search'

    # Default to greedy/local search
    return 'Greedy/Local Search'
```

### Step 4: Score Distribution Analysis

Analyze score distributions across paradigms:

```python
from collections import defaultdict

paradigm_scores = defaultdict(list)
for elite in elites:
    paradigm = classify_paradigm(elite['code'])
    paradigm_scores[paradigm].append(elite['score'])

for paradigm, scores in paradigm_scores.items():
    print(f"{paradigm}: min={min(scores):.2f}, max={max(scores):.2f}, count={len(scores)}")
```

### Step 5: Cluster Analysis (for PE validation)

To verify Punctuated Equilibrium is selecting diverse representatives:

```python
from sklearn.cluster import KMeans
import numpy as np

# Get behavior vectors
X = np.array([list(e['behavior'].values()) for e in elites])

# Cluster into n_clusters (matches PE config)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Find best elite per cluster (what PE would select)
representatives = []
for cluster_id in range(3):
    cluster_elites = [e for e, l in zip(elites, labels) if l == cluster_id]
    best = max(cluster_elites, key=lambda e: e['score'])
    representatives.append(best)
```

**Check if representatives are algorithmically distinct**, not just behaviorally distinct.

## Interpreting Results

### Good Diversity Indicators

- Multiple paradigm families with competitive scores
- High cell coverage (>50% of centroids occupied)
- Representatives from different paradigm families

### Warning Signs

- All top performers share same algorithmic paradigm
- Behavioral diversity but algorithmic monoculture
- Low cell coverage despite many elites (clustering)
- Noise inflation: behavior_noise > 0 causing artificial spread

### Noise Analysis

If `behavior_noise > 0` or `init_noise > 0` in config:

```python
# Re-extract behaviors with noise=0 to see true diversity
# Compare stored cell_id vs recalculated cell_id
# Difference indicates artificial niche occupation
```

## Common Paradigm Families

### 1. Binary Search + Bin Packing
Core: Binary search for optimal threshold/capacity, then greedy bin-pack assignment
- Fast convergence to good solutions
- Dominates when problem has clear threshold structure

### 2. Simulated Annealing
Core: Random moves with temperature-dependent acceptance
- Explores more of solution space
- Better at escaping local optima

### 3. Genetic/Evolutionary
Core: Population-based with crossover and mutation
- Good for combinatorial structure
- Higher computational cost

### 4. Greedy + Local Search
Core: Fast initial solution, iterative improvement
- Simple and fast
- May miss global optima

## Output Format

Provide analysis in this structure:

```
## Diversity Analysis Summary

**Archive Overview:**
- Total elites: N
- Unique cells: M
- Coverage: X%

**Paradigm Distribution:**
| Paradigm | Count | Best Score | Avg Score |
|----------|-------|------------|-----------|
| ...      | ...   | ...        | ...       |

**Top 10 Elites:**
| Rank | Score | Cell | Paradigm |
|------|-------|------|----------|
| ...  | ...   | ...  | ...      |

**Algorithmic Diversity Assessment:**
[Summary of whether archive contains genuinely different approaches]

**Recommendations:**
[Suggestions for improving diversity if needed]
```

## Running the Analysis Script

Use `uv run` to execute the analysis script with all dependencies:

```bash
uv run python skills/diversity-analysis/scripts/analyze_snapshot.py runs/YYYYMMDD_HHMMSS/snapshot.json --top 15 --tsne
```

Options:
- `--top N`: Show top N elites (default: 10)
- `--tsne`: Generate t-SNE visualization colored by paradigm

## Additional Resources

### Reference Files

For detailed analysis patterns:
- **`references/paradigm-markers.md`** - Extended paradigm classification rules

### Scripts

Utility scripts in `scripts/`:
- **`analyze_snapshot.py`** - Automated diversity analysis (run with `uv run python`)
