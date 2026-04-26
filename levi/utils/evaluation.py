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


def evaluate_prompt(prompt: str, score_fn: Callable[..., Any], inputs: Optional[list]) -> dict:
    """Runs in subprocess: score a prompt string directly."""
    try:
        accepts_1 = _accepts_n_positional_args(score_fn, 1)
        accepts_2 = _accepts_n_positional_args(score_fn, 2)

        if inputs is None:
            if accepts_1 is True:
                result = score_fn(prompt)
            elif accepts_2 is True:
                result = score_fn(prompt, [])
            else:
                result = score_fn(prompt)
        else:
            if accepts_2 is True:
                result = score_fn(prompt, inputs)
            elif accepts_1 is True:
                result = score_fn(prompt)
            else:
                result = score_fn(prompt, inputs)
    except MemoryError:
        return {"error": "MemoryError during scoring"}
    except Exception as e:
        return {"error": f"Scoring error: {e}"}

    return normalize_prompt_result(result)


def evaluate_bundle(
    bundle: dict[str, str], score_fn: Callable[..., Any], inputs: Optional[list]
) -> dict:
    """Runs in subprocess: score a prompt bundle dict directly."""
    try:
        accepts_1 = _accepts_n_positional_args(score_fn, 1)
        accepts_2 = _accepts_n_positional_args(score_fn, 2)

        if inputs is None:
            if accepts_1 is True:
                result = score_fn(bundle)
            elif accepts_2 is True:
                result = score_fn(bundle, [])
            else:
                result = score_fn(bundle)
        else:
            if accepts_2 is True:
                result = score_fn(bundle, inputs)
            elif accepts_1 is True:
                result = score_fn(bundle)
            else:
                result = score_fn(bundle, inputs)
    except MemoryError:
        return {"error": "MemoryError during scoring"}
    except Exception as e:
        return {"error": f"Scoring error: {e}"}

    return normalize_prompt_result(result)


def normalize_prompt_result(result: Any) -> dict:
    """Normalize prompt-evaluator outputs into Levi's metric dict shape."""
    if isinstance(result, (int, float)):
        return {"score": float(result)}

    if not isinstance(result, dict):
        return {"error": f"Prompt evaluator must return a number or dict, got {type(result).__name__}"}

    normalized: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, bool):
            normalized[key] = float(value)
        elif isinstance(value, (int, float)):
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


def coerce_score(result: dict) -> tuple[float | None, str | None]:
    """Extract a finite numeric score from an evaluation result."""
    try:
        score = float(result.get("score", 0.0))
    except (TypeError, ValueError):
        return None, f"Invalid score value: {result.get('score')!r}"
    if not math.isfinite(score):
        return None, f"Non-finite score value: {result.get('score')!r}"
    return score, None
