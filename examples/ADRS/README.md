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

# or without uv
python -m pip install -e ".[examples]"
python -m pip install -e ".[example-eplb]"
python -m pip install -e ".[example-llm-sql]"
```

Each problem directory uses the same entrypoint:

```bash
uv run --no-sync python run.py
```

If you installed with `pip` instead of `uv`, use `python run.py`.

## Problems

| Example | Problem | Published Budget | ADRS Data Needed |
|---------|---------|------------------|------------------|
| `cant_be_late/` | Schedule spot vs on-demand cloud instances to minimize cost under deadlines | $4.50 | Yes |
| `cant_be_late_multi/` | Same as above, but strategies can switch between regions | $5.00 | Yes |
| `cloudcast/` | Optimize broadcast topology across AWS, Azure, and GCP | $3.00 | Yes |
| `eplb/` | Place 64 MoE experts across 288 GPU slots to minimize KV cache pressure | $4.50 | Yes |
| `llm_sql/` | Reorder CSV columns to maximize prefix hit rate for LLM queries | $4.50 | No |
| `prism/` | Assign ML models to GPUs to minimize max KV cache pressure | $4.50 | No |
| `txn_scheduling/` | Order 100 transactions to minimize database makespan | $13.00 | No |

## Model Configuration

All examples use [LiteLLM](https://docs.litellm.ai/docs/providers) model identifiers. The published configs separate paradigm models (for creative exploration) from mutation models (for cheap, fast iteration):

- **Paradigm**: `openrouter/google/gemini-3-flash-preview`
- **Mutation**: `openrouter/xiaomi/mimo-v2-flash`, local Qwen 30B

The examples that use the local Qwen model assume `http://localhost:8000/v1`.
If your local server uses a different port or you want smaller budgets for a
smoke run, edit the corresponding lines in each `run.py`.

Set your API keys before running:

```bash
export OPENROUTER_API_KEY=your-key-here
export OPENAI_API_KEY=your-key-here  # if using OpenAI models
```
