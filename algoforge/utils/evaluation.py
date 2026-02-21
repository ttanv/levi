"""Canonical evaluation helpers used across all modules."""

import math
import types


def evaluate_code(code: str, score_fn, inputs: list, fn_name: str) -> dict:
    """Runs in subprocess: parse code, extract callable, call score_fn."""
    namespace = {}
    try:
        exec(code, namespace)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}
    except MemoryError:
        return {"error": "MemoryError during code execution"}

    fn = namespace.get(fn_name)
    if not isinstance(fn, types.FunctionType):
        return {"error": f"Function '{fn_name}' not found (got {type(fn).__name__})"}

    try:
        return score_fn(fn, inputs)
    except MemoryError:
        return {"error": "MemoryError during scoring"}
    except Exception as e:
        return {"error": f"Scoring error: {e}"}


def coerce_score(result: dict) -> tuple[float | None, str | None]:
    """Extract a finite numeric score from an evaluation result."""
    try:
        score = float(result.get("score", 0.0))
    except (TypeError, ValueError):
        return None, f"Invalid score value: {result.get('score')!r}"
    if not math.isfinite(score):
        return None, f"Non-finite score value: {result.get('score')!r}"
    return score, None
