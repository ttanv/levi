# Examples

The examples are grouped by how much setup they need.

## Quickstart — one file, runs in minutes

[`examples/quickstart/`](quickstart/) has three minimal single-file examples:

| File                                                | What it evolves | Needs                                  |
| --------------------------------------------------- | --------------- | -------------------------------------- |
| [`quickstart_api.py`](quickstart/quickstart_api.py) | code            | `OPENAI_API_KEY` (or other provider)   |
| [`quickstart_claude.py`](quickstart/quickstart_claude.py) | code       | `claude` CLI signed in                 |
| [`quickstart_prompts.py`](quickstart/quickstart_prompts.py) | a prompt | `OPENAI_API_KEY`                       |

Start here if you've never run LEVI before.

## Circle Packing — non-toy single-file problem

- Path: [`examples/circle_packing/`](circle_packing/)
- Self-contained (no external dataset), n=26 circles in a unit square.
- `run.py` defaults to OpenRouter + a local Qwen endpoint at `http://localhost:8000/v1`. Swap the `levi.LM(...)` for a hosted model id (e.g. `"openai/gpt-4o-mini"`) if you don't have a local server.
- `run_claude.py` runs the same problem via the Claude Code CLI.

## ADRS — paper benchmarks

[`examples/ADRS/`](ADRS/) contains the seven ADRS Leaderboard problems used in the LEVI paper. Each has a `run.py` configured for a small proposer model + a stronger paradigm-shift model.

Suggested picks:

- `ADRS/prism/` or `ADRS/llm_sql/` — simpler ADRS-style runs (no local server needed; `prism/` uses OpenRouter throughout).
- `ADRS/cant_be_late/` — exercises the `prompt_opt` sub-feature.
- `ADRS/cant_be_late_multi/` — full feature surface: custom `init`, punctuated equilibrium, `prompt_opt`.

See [`ADRS/README.md`](ADRS/README.md) for ADRS-specific setup and dataset requirements.

## Prompt-evolution benchmarks

Larger, dataset-backed prompt-evolution setups — closer in shape to GEPA's experimental harness than the quickstart:

- [`hotpotqa/`](hotpotqa/) — multi-hop QA (HuggingFace `hotpot_qa`).
- [`hover/`](hover/) — multi-hop fact verification with BM25 retrieval.
- [`pupa/`](pupa/) — privacy-preserving delegation (PAPILLON).
- [`ifbench/`](ifbench/) — instruction-following benchmark.

These pull HuggingFace datasets and assume an OpenRouter key for the proposer/judge models.
