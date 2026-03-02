# Examples

These examples are drawn from the [ADRS Leaderboard](https://github.com/cmu-db/ADRS-Leaderboard), a benchmark for LLM-guided algorithm discovery across systems optimization problems.

## Setup

Most examples require the ADRS Leaderboard dataset:

```bash
git clone https://github.com/cmu-db/ADRS-Leaderboard.git
export ADRS_EXAMPLE_DATA_ROOT=/path/to/ADRS-Leaderboard
```

Some examples have extra dependencies:

```bash
uv sync --extra examples        # all example deps
uv sync --extra example-eplb    # just EPLB (requires torch)
uv sync --extra example-llm-sql # just LLM SQL (requires pandas)
```

## Problems

| Example | Problem | Budget | Run |
|---------|---------|--------|-----|
| `cant_be_late/` | Schedule spot vs on-demand cloud instances to minimize cost under deadlines | $4.50 | `python run_levi_po.py` |
| `cant_be_late_multi/` | Same as above, but strategies can switch between regions | $5.00 | `python run_levi_po.py` |
| `cloudcast/` | Optimize broadcast topology across AWS, Azure, and GCP | $3.00 | `python run.py` |
| `eplb/` | Place 64 MoE experts across 288 GPU slots to minimize KV cache pressure | $4.50 | `python run_levi_po.py` |
| `llm_sql/` | Reorder CSV columns to maximize prefix hit rate for LLM queries | $4.50 | `python run_levi_po.py` |
| `prism/` | Assign ML models to GPUs to minimize max KV cache pressure | $4.50 | `python run_levi.py` |
| `txn_scheduling/` | Order 100 transactions to minimize database makespan | $13.00 | `python run_levi_po.py` |

## Model Configuration

All examples use [LiteLLM](https://docs.litellm.ai/docs/providers) model identifiers. The typical setup separates paradigm models (for creative exploration) from mutation models (for cheap, fast iteration):

- **Paradigm**: `openrouter/google/gemini-3-flash-preview`
- **Mutation**: `openrouter/xiaomi/mimo-v2-flash`, local Qwen 30B at `http://localhost:8001/v1`

Set your API keys before running:

```bash
export OPENROUTER_API_KEY=your-key-here
export OPENAI_API_KEY=your-key-here  # if using OpenAI models
```
