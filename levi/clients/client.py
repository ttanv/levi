"""Default LiteLLM-backed client implementation."""

import logging
import math
import os
from typing import Any, Optional

import litellm

from .base import BaseClient, ClientInput, ClientResult, ClientSpec, DEFAULT_TIMEOUT

litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)


def client_name(spec: ClientSpec) -> str:
    """Return the model identifier associated with a client spec."""
    if isinstance(spec, BaseClient):
        return spec.model
    return spec


def short_client_name(spec: ClientSpec) -> str:
    """Return the trailing segment of a model identifier for logs."""
    return client_name(spec).split("/")[-1]

def _normalize_prompt(prompt: ClientInput) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def _extract_text(model: str, response: Any) -> str:
    try:
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
    except Exception as error:
        raise ValueError(f"[{model}] Invalid completion schema") from error

    if content is None:
        content = ""
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or part.get("value")
                if isinstance(text, str):
                    parts.append(text)
        content = "\n".join(part for part in parts if part).strip()
    elif isinstance(content, dict):
        text = content.get("text") or content.get("content") or content.get("value")
        content = text if isinstance(text, str) else ""
    elif not isinstance(content, str):
        content = str(content)

    if not content:
        fallback_text = getattr(choice, "text", None)
        if isinstance(fallback_text, str):
            content = fallback_text

    return content


def _coerce_non_negative_float(value: Any) -> Optional[float]:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized) or normalized < 0.0:
        return None
    return normalized


def _usage_value(usage: Any, field: str) -> float:
    normalized = _coerce_non_negative_float(getattr(usage, field, None))
    return normalized or 0.0


def _cost_from_explicit_pricing(client: BaseClient, response: Any) -> Optional[float]:
    if (
        client.input_cost_per_token is None
        and client.output_cost_per_token is None
        and client.cache_read_input_token_cost is None
    ):
        return None

    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = _coerce_non_negative_float(getattr(prompt_details, "cached_tokens", None)) or 0.0
    cached_tokens = min(cached_tokens, prompt_tokens)
    uncached_prompt_tokens = max(0.0, prompt_tokens - cached_tokens)

    input_cost_per_token = client.input_cost_per_token or 0.0
    cache_read_input_token_cost = client.cache_read_input_token_cost
    if cache_read_input_token_cost is None:
        cache_read_input_token_cost = input_cost_per_token

    output_cost_per_token = client.output_cost_per_token or 0.0

    return (
        uncached_prompt_tokens * input_cost_per_token
        + cached_tokens * cache_read_input_token_cost
        + completion_tokens * output_cost_per_token
    )


def _extract_cost(client: BaseClient, response: Any) -> float:
    explicit_cost = _cost_from_explicit_pricing(client, response)
    if explicit_cost is not None:
        return explicit_cost

    usage = getattr(response, "usage", None)
    usage_cost = getattr(usage, "cost", None)
    if usage_cost is not None:
        try:
            normalized = float(usage_cost)
        except (TypeError, ValueError):
            normalized = float("nan")
        if math.isfinite(normalized) and normalized >= 0.0:
            return normalized

    try:
        return float(litellm.completion_cost(completion_response=response))
    except Exception:
        return 0.0


class Client(BaseClient):
    """Default generation client backed by LiteLLM."""

    def __init__(
        self,
        model: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        input_cost_per_token: Optional[float] = None,
        output_cost_per_token: Optional[float] = None,
        cache_read_input_token_cost: Optional[float] = None,
        **defaults: Any,
    ):
        super().__init__(
            model,
            timeout=timeout,
            input_cost_per_token=input_cost_per_token,
            output_cost_per_token=output_cost_per_token,
            cache_read_input_token_cost=cache_read_input_token_cost,
            **defaults,
        )

    async def acompletion(self, prompt: ClientInput, **kwargs: Any) -> ClientResult:
        request: dict[str, Any] = dict(self.defaults)
        request.update(kwargs)
        request["model"] = self.model
        request["messages"] = _normalize_prompt(prompt)
        request.setdefault("timeout", self.timeout)
        request = {key: value for key, value in request.items() if value is not None}

        try:
            response = await litellm.acompletion(**request)
        except Exception as error:
            message = str(error)
            if "LLM Provider NOT provided" not in message:
                raise

            retry_request = dict(request)
            retry_request["custom_llm_provider"] = "openai"
            if "api_base" not in retry_request:
                resolved_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
                if resolved_base:
                    retry_request["api_base"] = resolved_base
                else:
                    logger.warning(
                        "[%s] No api_base found for provider fallback; retrying against the default OpenAI endpoint",
                        self.model,
                    )

            if "api_key" not in retry_request:
                retry_request["api_key"] = os.getenv("OPENAI_API_KEY") or "local-no-key-required"

            response = await litellm.acompletion(**retry_request)

        cost = _extract_cost(self, response)
        text = _extract_text(self.model, response)
        return ClientResult(text=text, cost=cost)


class ClientResolver:
    """Small run-local cache that resolves string model specs into client instances."""

    def __init__(
        self,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self._clients: dict[str, BaseClient] = {}
        self.configure(temperature=temperature, max_tokens=max_tokens, timeout=timeout)

    def configure(
        self,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._defaults: dict[str, Any] = {}
        if temperature is not None:
            self._defaults["temperature"] = temperature
        if max_tokens is not None:
            self._defaults["max_tokens"] = max_tokens
        self._timeout = timeout
        self._clients.clear()

    def resolve(self, spec: ClientSpec) -> BaseClient:
        if isinstance(spec, BaseClient):
            return spec

        client = self._clients.get(spec)
        if client is None:
            client = Client(spec, timeout=self._timeout, **self._defaults)
            self._clients[spec] = client
        return client

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()


def resolve_client(spec: ClientSpec, **kwargs: Any) -> BaseClient:
    """Resolve a client spec without caching."""
    if isinstance(spec, BaseClient):
        return spec
    return Client(spec, **kwargs)
