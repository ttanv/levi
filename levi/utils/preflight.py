"""Pre-flight checks run before the evolutionary pipeline starts.

Catches the most common first-run failure — missing API key — and turns it
into a clear ``EnvironmentError`` instead of letting litellm fail mid-run.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from ..clients.base import BaseClient, ClientSpec
from ..clients.lm import LM

PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "perplexity": "PERPLEXITYAI_API_KEY",
    "xai": "XAI_API_KEY",
    "fireworks_ai": "FIREWORKS_API_KEY",
}


def _required_env_var(spec: ClientSpec) -> str | None:
    """Return the env var name a model spec needs, or None if no check applies.

    Returns None for: non-LM BaseClient subclasses (CLI clients carry their own
    auth), LMs with an explicit ``api_key`` or ``api_base`` (caller has handled
    auth), and unrecognized provider prefixes.
    """
    if isinstance(spec, BaseClient) and not isinstance(spec, LM):
        return None

    model_id: str
    if isinstance(spec, LM):
        if "api_key" in spec.defaults or "api_base" in spec.defaults:
            return None
        model_id = spec.model
    elif isinstance(spec, str):
        model_id = spec
    else:
        return None

    prefix = model_id.split("/", 1)[0].lower() if "/" in model_id else ""
    return PROVIDER_ENV_VARS.get(prefix)


def check_api_keys(model_specs: Iterable[ClientSpec]) -> None:
    """Raise ``EnvironmentError`` if any model spec needs an env var that isn't set.

    Specs whose provider is unrecognized are skipped silently — we'd rather miss
    a case than block a working configuration.
    """
    missing: dict[str, set[str]] = {}
    for spec in model_specs:
        env_var = _required_env_var(spec)
        if env_var is None or os.getenv(env_var):
            continue
        model_id = spec.model if isinstance(spec, BaseClient) else str(spec)
        missing.setdefault(env_var, set()).add(model_id)

    if not missing:
        return

    lines = ["Missing API key(s) required by the configured model(s):"]
    for env_var, models in sorted(missing.items()):
        models_str = ", ".join(sorted(models))
        lines.append(f"  - {env_var}  (needed for: {models_str})")
    lines.append("Set the variable in your environment (e.g. `export OPENAI_API_KEY=...`) and re-run.")
    raise EnvironmentError("\n".join(lines))
