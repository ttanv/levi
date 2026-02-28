# Levi

**Better LLM Optimization for the Price of a Cup of Coffee**

[![CI](https://github.com/ttanv/algoforge/actions/workflows/ci.yml/badge.svg)](https://github.com/ttanv/algoforge/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Levi is an LLM-guided evolutionary framework for discovering algorithms, heuristics, and optimized code. Point it at a scoring function and a seed program, set a dollar budget, and walk away.

## Why Levi

Existing frameworks  couple performance tightly to model capability. Drop to a smaller model and results degrade sharply. Levi decouples the two by making diversity an architectural concern rather than a model concern, and by matching model capacity to task demand: cheap models for refinement, expensive models only for periodic creative leaps. Set a dollar budget and Levi spends it well.

**$4.50 matches what other frameworks need $15–30 and frontier models to achieve.** Highest scores on the [ADRS benchmark](https://github.com/cmu-db/ADRS-Leaderboard) across all frameworks. See [detailed results](https://ttanv.github.io/levi).

## Quickstart

```bash
pip install levi
# or from source
git clone https://github.com/ttanv/algoforge.git && cd algoforge
uv sync
```

```python
import levi

result = levi.evolve_code(
    "Optimize bin packing to minimize wasted space",
    function_signature="def pack(items, bin_capacity):",
    seed_program="def pack(items, bin_capacity):\n    return [[item] for item in items]",
    score_fn=lambda fn, _: {"score": max(0, 100 - sum(10 - sum(b) for b in fn([4,8,1,4,2,1], 10)) * 10)},
    model="openai/gpt-4o-mini",
    budget_dollars=2.0,
)

print(result.best_program.code)
print(f"Best score: {result.best_score}")
```

## How It Works

1. **Problem definition** — A natural-language description, a function signature, a seed program, and a scoring function.
2. **Initialization** — Levi generates structurally diverse seed variants and uses them to set up a behavioral archive that maintains diversity across fundamentally different solution strategies.
3. **Evolution loop** — An async pipeline samples parents from the archive, mutates them via LLM calls, evaluates the results, and inserts improvements back. Most mutations are routed to cheap workhorse models for local refinements.
4. **Paradigm shifts** — Periodically, a stronger model is given representative solutions from across the archive and asked to propose structurally different approaches — new algorithmic families rather than incremental improvements.
5. **Budget-aware stopping** — Levi tracks spend in real time and stops when the dollar, evaluation, or time budget is exhausted. No guessing how many iterations to run.

Key concepts:
- **Seed program**: A working (but suboptimal) starting solution.
- **Score function**: Returns `{"score": float}` (higher is better). Can include additional keys for sub-metrics.
- **Behavioral archive**: Programs are placed into niches defined by code-structure features, preventing premature convergence regardless of model capability.
- **Stratified model allocation**: Cheap models for the volume work, expensive models for creative leaps.

## Configuration

### Model specification

```python
# Single model for everything
result = levi.evolve_code(..., model="openai/gpt-4o-mini")

# Separate paradigm (creative) and mutation (workhorse) models
result = levi.evolve_code(
    ...,
    paradigm_model="openai/gpt-4o",
    mutation_model="openai/gpt-4o-mini",
)

# Multiple mutation models (round-robin)
result = levi.evolve_code(
    ...,
    paradigm_model="openai/gpt-4o",
    mutation_model=["openai/gpt-4o-mini", "openrouter/qwen/qwen3-30b"],
)
```

Models are specified using [LiteLLM](https://docs.litellm.ai/docs/providers) format (`provider/model-name`).

### Budget

```python
result = levi.evolve_code(..., budget_dollars=5.0)   # cost cap
result = levi.evolve_code(..., budget_evals=200)      # evaluation cap
result = levi.evolve_code(..., budget_seconds=3600)   # time cap
```

### Advanced configuration

```python
result = levi.evolve_code(
    ...,
    pipeline=levi.PipelineConfig(n_llm_workers=8, n_eval_processes=8),
    behavior=levi.BehaviorConfig(features=["cyclomatic_complexity", "branch_count"]),
    punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(enabled=True, interval=5),
    prompt_opt=levi.PromptOptConfig(enabled=True),
)
```

See `levi.LeviConfig` for the full list of configuration options.

## Examples

Seven worked examples from the [ADRS Leaderboard](https://github.com/UCB-ADRS/ADRS-Leaderboard) benchmark — cloud scheduling, GPU placement, broadcast optimization, and more. See [`examples/README.md`](examples/README.md) for setup and details.


