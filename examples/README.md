# Examples

## Circle Packing

Self-contained optimization example with no external dataset:

- Path: `examples/circle_packing/`
- Run: `cd examples/circle_packing && uv run --no-sync python run.py`
- Without `uv`: `cd examples/circle_packing && python run.py`

The example assumes the local Qwen endpoint is `http://localhost:8000/v1`.
If your local server uses a different port or you want a smaller budget, edit
the `local_endpoints`, `paradigm_model`, `mutation_model`, or `budget_dollars`
lines in `run.py`.

## ADRS Examples

Additional ADRS benchmark examples live under:

- `examples/ADRS/`

Each example directory uses `run.py` as its entrypoint.

See `examples/ADRS/README.md` for ADRS-specific setup and dataset requirements.
