# Examples

## Circle Packing

Self-contained optimization example with no external dataset:

- Path: `examples/circle_packing/`
- Run: `cd examples/circle_packing && uv run --no-sync python run.py`
- Without `uv`: `cd examples/circle_packing && python run.py`

Use these optional environment variables to customize models/budget:

- `LEVI_LOCAL_ENDPOINT` (default: `http://localhost:8001/v1`)
- `LEVI_BUDGET_DOLLARS` (default: `1.0`)

## ADRS Examples

Additional ADRS benchmark examples live under:

- `examples/ADRS/`

Each example directory uses `run.py` as its entrypoint.

See `examples/ADRS/README.md` for ADRS-specific setup and dataset requirements.
