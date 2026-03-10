# Circle Packing Example (Comparable Setup)

This uses the common benchmark setup:
- `n = 26` circles
- unit square `[0,1] x [0,1]`
- objective: maximize `sum(radii)`

Candidate function API:

```python
def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # returns (centers, radii, sum_radii)
```

The evaluator enforces boundary and non-overlap constraints. Invalid packings
get score `0`.

If you installed LEVI with `pip` instead of `uv`, replace `uv run --no-sync python run.py`
below with `python run.py`.

## Run

```bash
cd examples/circle_packing
uv run --no-sync python run.py
```

Quick smoke test:

```bash
cd examples/circle_packing
LEVI_BUDGET_DOLLARS=0.05 uv run --no-sync python run.py
```

Optional overrides:

```bash
export LEVI_BUDGET_DOLLARS=3.0
export LEVI_LOCAL_ENDPOINT=http://localhost:8001/v1
uv run --no-sync python run.py
```

Default model is ADRS-style local Qwen:
- `Qwen/Qwen3-30B-A3B-Instruct-2507`
- endpoint: `http://localhost:8001/v1`

## Quick Sanity Check (Seed Score)

```bash
cd examples/circle_packing
python - <<'PY'
from problem import score_fn, SEED_PROGRAM

scope = {}
exec(SEED_PROGRAM, scope)
run_packing = scope["run_packing"]
print(score_fn(run_packing))
PY
```
