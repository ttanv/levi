"""Canonical evaluation helpers used across all modules."""

import inspect
import math
import types
from collections.abc import Callable
from typing import Any, Optional


def _accepts_n_positional_args(fn: Callable[..., Any], n: int) -> Optional[bool]:
    """Best-effort check whether `fn` can accept `n` positional args."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None

    params = list(sig.parameters.values())
    positional_params = [
        p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required_positional = [p for p in positional_params if p.default is inspect.Parameter.empty]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

    min_args = len(required_positional)
    max_args = float("inf") if has_varargs else len(positional_params)
    return bool(min_args <= n <= max_args)


def _call_score_fn(score_fn: Callable[..., dict], fn: Callable[..., Any], inputs: Optional[list]) -> dict:
    """
    Call score_fn with either signature:
    - score_fn(fn)
    - score_fn(fn, inputs)
    """
    accepts_1 = _accepts_n_positional_args(score_fn, 1)
    accepts_2 = _accepts_n_positional_args(score_fn, 2)

    if inputs is None:
        if accepts_1 is True:
            return score_fn(fn)
        if accepts_2 is True:
            return score_fn(fn, [])
        # Fallback when arity detection is unavailable.
        return score_fn(fn)

    if accepts_2 is True:
        return score_fn(fn, inputs)
    if accepts_1 is True:
        return score_fn(fn)
    # Fallback when arity detection is unavailable.
    return score_fn(fn, inputs)


def evaluate_code(code: str, score_fn: Callable[..., dict], inputs: Optional[list], fn_name: str) -> dict:
    """Runs in subprocess: parse code, extract callable, call score_fn."""
    namespace = {}
    try:
        exec(code, namespace)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}
    except MemoryError:
        return {"error": "MemoryError during code execution"}

    namespace["__source_code__"] = code

    fn = namespace.get(fn_name)
    if not isinstance(fn, types.FunctionType):
        return {"error": f"Function '{fn_name}' not found (got {type(fn).__name__})"}

    try:
        return _call_score_fn(score_fn, fn, inputs)
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
