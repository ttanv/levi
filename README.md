# CEvolve

**CEvolve** is an LLM-guided evolutionary method for discovering algorithms, heuristics, and optimized code. Existing approaches like OpenEvolve and GEPA use complex architectures with multi-island populations, migration protocols, and novelty bonuses, yet still struggle with premature convergence. AlgoForge takes a simpler approach: measure diversity using deterministic code structure features via CVT-MAP-Elites, which removes the need for much of this architectural complexity. The result is a single-island system that outperforms multi-island competitors at a fraction of the compute cost. CEvolve targets systems optimization, prompt engineering, and algorithm discovery.

## Getting Started

Install core AlgoForge dependencies:

```bash
uv sync
```

Install optional extras only when you need them:

```bash
# EPLB example (requires torch): examples/eplb/problem.py
uv sync --extra example-eplb

# LLM SQL example
uv sync --extra example-llm-sql

# Install all example-related dependencies at once
uv sync --extra examples
```
