# Extended Paradigm Classification Markers

This reference provides detailed code patterns for classifying algorithms into paradigm families.

## Binary Search Family

### Pure Binary Search
```python
# Markers
while lo < hi:        # or while left < right
mid = (lo + hi) // 2  # or mid = left + (right - left) // 2
lo = mid + 1          # narrowing from below
hi = mid              # narrowing from above
```

### Binary Search + Feasibility Check
```python
# Pattern: Binary search outer loop with greedy inner check
def is_feasible(threshold):
    # Greedy assignment checking if threshold works
    ...
    return success

lo, hi = min_val, max_val
while lo < hi:
    mid = (lo + hi) // 2
    if is_feasible(mid):
        hi = mid
    else:
        lo = mid + 1
```

### Binary Search + Bin Packing
Combines binary search for optimal capacity/threshold with bin-packing assignment:
```python
# Outer: binary search for KVPR threshold
# Inner: First-Fit Decreasing (FFD) or Best-Fit Decreasing (BFD)
sorted_items = sorted(items, key=lambda x: x.size, reverse=True)
for item in sorted_items:
    # Assign to first/best bin that fits
```

## Simulated Annealing Family

### Classic SA
```python
# Key markers
temperature = initial_temp
while temperature > min_temp:
    # Generate neighbor solution
    neighbor = perturb(current)
    delta = cost(neighbor) - cost(current)

    # Acceptance probability
    if delta < 0 or random.random() < math.exp(-delta / temperature):
        current = neighbor

    temperature *= cooling_rate  # Often 0.95-0.99
```

### SA Variants
- **Fast SA**: Cauchy distribution for perturbations
- **Adaptive SA**: Temperature adjusts based on acceptance rate
- **Threshold Accepting**: Accept if delta < threshold (no randomness)

### Markers
- `temperature` or `temp` or `T` variable
- `math.exp(-` or `exp(-delta`
- `cooling` or `* 0.9` patterns
- `random() <` probability acceptance

## Genetic/Evolutionary Family

### Genetic Algorithm
```python
# Population-based markers
population = [random_solution() for _ in range(pop_size)]
for generation in range(n_generations):
    # Selection
    parents = tournament_select(population)

    # Crossover
    offspring = crossover(parent1, parent2)

    # Mutation
    if random.random() < mutation_rate:
        mutate(offspring)

    population = select_survivors(population + offspring)
```

### Evolution Strategy
```python
# (mu, lambda) or (mu + lambda) selection
parents = population[:mu]
offspring = [mutate(random.choice(parents)) for _ in range(lambda_)]
```

### Markers
- `population` variable
- `crossover` or `recombine` functions
- `mutation` or `mutate` functions
- `fitness` scoring
- `generation` loops
- `tournament` or `roulette` selection

## Tabu Search Family

### Classic Tabu Search
```python
tabu_list = []  # or set() or deque(maxlen=tenure)
tabu_tenure = 7  # How long moves stay forbidden

current = initial_solution()
best = current

for iteration in range(max_iterations):
    neighbors = get_neighbors(current)

    # Filter out tabu moves (unless aspiration)
    allowed = [n for n in neighbors if n.move not in tabu_list]

    # Select best non-tabu neighbor
    next_sol = min(allowed, key=cost)

    # Update tabu list
    tabu_list.append(next_sol.move)
    if len(tabu_list) > tabu_tenure:
        tabu_list.pop(0)

    current = next_sol
```

### Markers
- `tabu_list` or `tabu` variable
- `tenure` parameter
- Move history tracking
- Aspiration criteria (override tabu if better than best)

## Greedy Family

### Simple Greedy
```python
# Single pass, local decisions
solution = []
for item in items:
    best_choice = min(options, key=score_function)
    solution.append(best_choice)
```

### Sorted Greedy
```python
# Sort first, then greedy assign
sorted_items = sorted(items, key=priority, reverse=True)
for item in sorted_items:
    assign_to_best_available(item)
```

### Variants
- **First-Fit (FF)**: Assign to first valid option
- **Best-Fit (BF)**: Assign to option minimizing waste
- **Worst-Fit (WF)**: Assign to option with most remaining capacity

### Markers
- Single loop over items
- No backtracking or undo
- Local min/max selection
- `sorted()` preprocessing

## Local Search Family

### Hill Climbing
```python
current = initial_solution()
while True:
    neighbors = get_neighbors(current)
    best_neighbor = min(neighbors, key=cost)

    if cost(best_neighbor) >= cost(current):
        break  # Local optimum reached

    current = best_neighbor
```

### Iterated Local Search (ILS)
```python
# Local search + perturbation + restart
current = local_search(initial_solution())
best = current

for iteration in range(max_iterations):
    # Perturb to escape local optimum
    perturbed = perturb(current)

    # Apply local search
    improved = local_search(perturbed)

    # Accept/reject
    if accept(improved, current):
        current = improved
        if cost(improved) < cost(best):
            best = improved
```

### Variable Neighborhood Search (VNS)
```python
k = 1  # Neighborhood index
while k <= k_max:
    # Shake: random from k-th neighborhood
    x_prime = shake(current, k)

    # Local search
    x_improved = local_search(x_prime)

    if cost(x_improved) < cost(current):
        current = x_improved
        k = 1  # Restart from first neighborhood
    else:
        k += 1  # Move to next neighborhood
```

### Markers
- `neighbor` or `neighbors` generation
- `improve` or `improvement` loops
- Convergence checks
- `while True` with break conditions

## Dynamic Programming Family

### Memoization
```python
memo = {}
def solve(state):
    if state in memo:
        return memo[state]

    # Compute result
    result = ...

    memo[state] = result
    return result
```

### Tabulation
```python
dp = [[0] * m for _ in range(n)]

# Fill base cases
dp[0][0] = base_value

# Fill table
for i in range(1, n):
    for j in range(1, m):
        dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + cost[i][j]
```

### Markers
- `memo` or `cache` dictionary
- `dp` or `table` array
- `@lru_cache` decorator
- Subproblem indexing

## Constraint Programming Family

### Backtracking
```python
def solve(partial_solution, index):
    if index == n:
        return partial_solution  # Complete solution

    for choice in domain[index]:
        if is_valid(partial_solution, choice):
            partial_solution.append(choice)
            result = solve(partial_solution, index + 1)
            if result:
                return result
            partial_solution.pop()  # Backtrack

    return None
```

### Branch and Bound
```python
def branch_and_bound(node, best_cost):
    if is_complete(node):
        return cost(node)

    # Pruning: lower bound exceeds best
    if lower_bound(node) >= best_cost:
        return best_cost

    for child in expand(node):
        best_cost = min(best_cost, branch_and_bound(child, best_cost))

    return best_cost
```

### Markers
- `backtrack` function or recursion with undo
- `bound` or `lower_bound` functions
- `prune` conditions
- Recursive exploration with constraints

## Hybrid Detection

Many elite algorithms combine paradigms. Detection order:

1. Check for outer binary search loop
2. Check for SA temperature/acceptance
3. Check for population-based evolution
4. Check for tabu memory
5. Check for DP memoization
6. Default to greedy/local search

Report hybrids as "Binary Search + [inner paradigm]" when applicable.

## False Positive Handling

### "mutation" keyword
- In genetic context: True GA
- In isolation (variable naming): Likely NOT GA
- Check for: population, crossover, fitness

### "memo" keyword
- True DP: Used for subproblem caching
- False positive: Just a memo/note variable
- Check for: recursive structure, state indexing

### Loop nesting
- High nesting doesn't imply specific paradigm
- Check actual loop purpose (enumeration vs optimization)
