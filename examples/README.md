# Examples

## Circle Packing (Current)

Self-contained optimization example with no external dataset:

- Path: `examples/circle_packing/`
- Run: `cd examples/circle_packing && uv run --no-sync python run_levi.py`

Use these optional environment variables to customize models/budget:

- `LEVI_LOCAL_ENDPOINT` (default: `http://localhost:8001/v1`)
- `LEVI_BUDGET_DOLLARS` (default: `1.0`)

## ADRS Archive (Previous Runs)

All previous ADRS examples were moved under:

- `examples/ADRS/`

See `examples/ADRS/README.md` for ADRS-specific setup and dataset requirements.
