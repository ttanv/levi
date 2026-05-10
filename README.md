<p align="center">
  <img src="assets/logos/levi_logo_dark.svg#gh-dark-mode-only" width="25%" alt="LEVI" />
  <img src="assets/logos/levi_logo_light.svg#gh-light-mode-only" width="25%" alt="LEVI" />
</p>

<p align="center"><strong>AlphaEvolve Performance for a Fraction of the Cost</strong></p>

<p align="center">
  <a href="https://github.com/ttanv/levi/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/ttanv/levi/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://ttanv.github.io/levi"><img src="https://img.shields.io/badge/docs-ttanv.github.io%2Flevi-orange" alt="Docs"></a>
</p>

---

LEVI is an LLM-guided evolutionary framework for **code** and **prompts**. Point it at a scoring function and a budget — LEVI evolves the artifact for you, using API models, a local server, or your Claude Code / Codex CLI subscription.

**$4.50 improves on what other frameworks need $15-30 and frontier models to achieve** across a [variety of problems](https://ucbskyadrs.github.io/), at a fraction of the cost.

## Why LEVI

Existing frameworks couple performance tightly to model capability. Drop to a smaller model and results degrade sharply. LEVI decouples the two by making **diversity an architectural concern** rather than a model concern, and by matching model capacity to task demand.

Cheap models handle the bulk of mutation work. A behavioral archive keeps structurally different strategies alive, preventing premature convergence. Periodic paradigm shifts from a stronger model inject genuinely new ideas. The result: you spend less and get more.


<p align="center">
  <img src="assets/plots/txn_scheduling.png" width="49%" />
  <img src="assets/plots/cant_be_late.png" width="49%" />
</p>
<p align="center"><em>LEVI converges faster and scores higher than baselines on controlled equal-budget comparisons (same model, 750 evals).</em></p>

## Quickstart

```bash
# Install uv first: https://docs.astral.sh/uv/getting-started/installation/
git clone https://github.com/ttanv/levi.git
cd levi
uv sync
```

Pick whichever path matches what you have access to — each is a single self-contained file under [`examples/quickstart/`](examples/quickstart/), runs in a couple of minutes, costs a few cents (or nothing on the CLI path):

| You have…                             | Run                                            | Evolves   |
| ------------------------------------- | ---------------------------------------------- | --------- |
| an API key (OpenAI / Anthropic / …)   | `uv run python examples/quickstart/quickstart_api.py`     | code      |
| a Claude Code / Codex CLI subscription | `uv run python examples/quickstart/quickstart_claude.py`  | code      |
| an API key, and you want to tune prompts | `uv run python examples/quickstart/quickstart_prompts.py` | prompts   |

Set `OPENAI_API_KEY` (or change `MODEL` at the top of the file to another [litellm provider](https://docs.litellm.ai/docs/providers) and set the matching key) before running the API quickstarts.

A minimal LEVI program looks like this:

```python
from collections import Counter
import levi

def score_fn(pack, inputs):
    scores = []
    for items in inputs:
        bins = pack(list(items), 10)
        if any(sum(b) > 10 for b in bins) or Counter(x for b in bins for x in b) != Counter(items):
            return {"score": 0.0}
        scores.append(100.0 * sum(sum(b) for b in bins) / (len(bins) * 10))
    return {"score": sum(scores) / len(scores)}

if __name__ == "__main__":
    result = levi.evolve_code(
        "Pack items into bins of capacity 10, minimizing wasted space.",
        function_signature="def pack(items: list[int], capacity: int) -> list[list[int]]:",
        score_fn=score_fn,
        inputs=[[4, 8, 1, 4, 2, 1], [9, 2, 3, 7, 8, 1, 4]],
        model="openai/gpt-4o-mini",
        budget_dollars=0.10,
    )
    print(result.best_score, result.best_program)
```

The `if __name__ == "__main__":` guard matters — LEVI runs evaluations in subprocesses (`spawn` on macOS / Windows).

## Going further

- `examples/quickstart/` — the three single-file starters above.
- `examples/circle_packing/` — n=26 circle packing benchmark; the simplest non-toy problem.
- `examples/ADRS/` — seven ADRS Leaderboard problems used in the paper. Most use a cheap proposer model via OpenRouter (or a local Qwen server) plus stronger paradigm-shift calls. See `examples/ADRS/README.md` for setup.
- `examples/hotpotqa/`, `examples/hover/`, `examples/pupa/`, `examples/ifbench/` — prompt-evolution benchmarks comparing against GEPA.

## Results

LEVI holds the **highest average score (76.5)** across all seven [ADRS Leaderboard](https://ucbskyadrs.github.io/) problems, ahead of GEPA (71.9), OpenEvolve (70.6), and ShinkaEvolve (67.4). Six of the seven problems were solved on a **$4.50 budget**.

| Problem | LEVI | Best Other Framework | Saving |
|---------|------|----------------------|--------|
| Spot Single-Reg | **51.7** | GEPA 51.4 | 6.7x cheaper |
| Spot Multi-Reg | **72.4** | OpenEvolve 66.7 | 5.6x cheaper |
| LLM-SQL | **78.3** | OpenEvolve 72.5 | 4.4x cheaper |
| Cloudcast | **100.0** | GEPA 96.6 | 3.3x cheaper |
| Prism | **87.4** | GEPA / OpenEvolve / ShinkaEvolve 87.4 | 3.3x cheaper |
| EPLB | **74.6** | GEPA 70.2 | 3.3x cheaper |
| Txn Scheduling | **71.1** | OpenEvolve 70.0 | 1.5x cheaper |

<p align="center">
  <img src="assets/plots/circle_packing_best.png#gh-dark-mode-only" width="50%" alt="Circle Packing" />
  <img src="assets/plots/circle_packing_best_light.png#gh-light-mode-only" width="50%" alt="Circle Packing" />
</p>

LEVI scored **2.6359+ packing density** on the n=26 circle packing benchmark, with a local model handling the majority of mutations. See [`examples/circle_packing`](examples/circle_packing) for the full setup.

For advanced routing, pass a `levi.LM(...)` directly:

```python
local_qwen = levi.LM(
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    api_base="http://localhost:8000/v1",
    api_key="unused",
    input_cost_per_token=0.0000001,
    output_cost_per_token=0.0000004,
)
```

## How It Works

1. **Seed & score.** You provide a starting program and a scoring function. LEVI generates diverse variants to populate a behavioral archive.
2. **Evolve.** Cheap models mutate and refine solutions in parallel. The behavioral archive keeps structurally different strategies alive, preventing convergence.
3. **Paradigm shifts.** Periodically, a stronger model proposes entirely new algorithmic approaches based on the archive's best ideas.
4. **Budget stops.** LEVI tracks spend in real time and stops when your dollar, evaluation, or time cap is hit.

Read more in the [full writeup](https://ttanv.github.io/levi/docs#how-it-works).

## Further Reading

- [Documentation](https://ttanv.github.io/levi) — API reference, configuration, architecture details, and ablations.
- [LEVI: LLM-Guided Evolutionary Search Needs Better Harnesses, Not Bigger Models](https://ucbskyadrs.github.io/blog/levi/) — The full blog post on the ADRS site.

## Citation

If you use LEVI in your research, please cite:

```bibtex
@software{tanveer2026levi,
  title  = {LEVI: LLM-Guided Evolutionary Search Needs Better Harnesses, Not Bigger Models},
  author = {Tanveer, Temoor},
  url    = {https://github.com/ttanv/levi},
  year   = {2026}
}
```

Contact: ttanveer@alumni.cmu.edu
