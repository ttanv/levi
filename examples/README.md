# Examples

Start with `examples/circle_packing/` if you are new to LEVI. It is the
smallest end-to-end example in the repo.

## Circle Packing

Self-contained optimization example with no external dataset:

- Setup: `uv sync`
- Path: `examples/circle_packing/`
- Run: `cd examples/circle_packing && uv run python run.py`

The example assumes the local Qwen endpoint is `http://localhost:8000/v1`.
If your local server uses a different port or you want a smaller budget, edit
the `local_endpoints`, `paradigm_model`, `mutation_model`, or `budget_dollars`
lines in `run.py`.

## ADRS Examples

Additional ADRS benchmark examples live under:

- `examples/ADRS/`

Each example directory uses `run.py` as its entrypoint.
Install the repo first with `uv sync`.

Suggested picks:

- `examples/ADRS/prism/` or `examples/ADRS/llm_sql/` for a simpler ADRS-style run
- `examples/ADRS/cant_be_late/` if you want to try `prompt_opt`
- `examples/ADRS/cant_be_late_multi/` if you want the most feature-complete example with custom `init`, punctuated equilibrium, and `prompt_opt`

See `examples/ADRS/README.md` for ADRS-specific setup and dataset requirements.
