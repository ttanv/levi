"""Unified LLM Client — all calls routed through litellm."""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import litellm

litellm.suppress_debug_info = True

from .exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


@dataclass
class CompletionResponse:
    """Response from an LLM completion."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cost: float
    raw_response: Optional[Any] = None


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _wrap_litellm_error(model: str, error: Exception) -> Exception:
    """Map litellm exceptions into internal LLM exception types."""
    message = str(error).lower()
    error_type = type(error).__name__.lower()
    tag = f"[{model}] {error}"

    if "timeout" in message or "timeout" in error_type:
        return LLMTimeoutError(tag)
    if "rate limit" in message or "ratelimit" in message or "toomanyrequests" in error_type:
        return LLMRateLimitError(tag)
    if "authentication" in message or "unauthorized" in message or "forbidden" in message:
        return LLMAuthenticationError(tag)
    if "connection" in message or "network" in message:
        return LLMConnectionError(tag)
    return LLMResponseError(tag)


# ---------------------------------------------------------------------------
# Config & Client
# ---------------------------------------------------------------------------


@dataclass
class UnifiedLLMClientConfig:
    """Configuration for the unified LLM client."""

    temperature: float = 0.8
    max_tokens: int = 16384
    timeout: float = 300.0


class UnifiedLLMClient:
    """
    Unified LLM client — all calls routed through litellm.

    Features:
    - Single provider (litellm handles cloud + local routing)
    - Cost tracking
    - Both async and sync interfaces
    """

    def __init__(self, config: Optional[UnifiedLLMClientConfig] = None):
        self._config = config or UnifiedLLMClientConfig()
        self._total_cost = 0.0

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        stop: Optional[list[str]] = None,
        **extras,
    ) -> CompletionResponse:
        """Async completion via litellm."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._config.max_tokens,
            "timeout": timeout if timeout is not None else self._config.timeout,
        }
        if stop:
            kwargs["stop"] = stop
        kwargs.update(extras)

        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as e:
            # Some local OpenAI-compatible servers (e.g., vLLM) expose model IDs
            # like "Qwen/..." that need explicit provider selection in LiteLLM.
            # Retry once with provider forced to OpenAI while preserving model ID.
            msg = str(e)
            if "LLM Provider NOT provided" in msg:
                try:
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["custom_llm_provider"] = "openai"
                    # Prefer registered local endpoint details from litellm.register_model().
                    local_params = getattr(litellm, "model_cost", {}).get(model, {}).get("litellm_params", {})
                    if "api_base" not in retry_kwargs:
                        resolved_base = (
                            local_params.get("api_base") or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
                        )
                        if resolved_base:
                            retry_kwargs["api_base"] = resolved_base
                        else:
                            logger.warning(
                                "[%s] No api_base found for provider fallback — "
                                "the retry will target the default OpenAI endpoint.",
                                model,
                            )
                    # OpenAI-compatible local servers (e.g., vLLM) may ignore key,
                    # but the OpenAI client path still requires a non-empty value.
                    if "api_key" not in retry_kwargs:
                        retry_kwargs["api_key"] = (
                            local_params.get("api_key") or os.getenv("OPENAI_API_KEY") or "local-no-key-required"
                        )
                    logger.debug(
                        "[%s] Retrying with custom_llm_provider='openai' (api_base=%s)",
                        model,
                        retry_kwargs.get("api_base", "<default>"),
                    )
                    response = await litellm.acompletion(**retry_kwargs)
                except Exception as retry_error:
                    raise _wrap_litellm_error(model, retry_error) from retry_error
            else:
                raise _wrap_litellm_error(model, e) from e

        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        usage = getattr(response, "usage", None)
        if usage is None:
            raise LLMResponseError(f"[{model}] Missing usage field in response")

        try:
            content = response.choices[0].message.content
        except Exception as e:
            raise LLMResponseError(f"[{model}] Invalid completion schema") from e

        self._total_cost += cost

        return CompletionResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model=model,
            cost=cost,
            raw_response=response,
        )

    def completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        stop: Optional[list[str]] = None,
        **extras,
    ) -> CompletionResponse:
        """Sync wrapper for acompletion."""
        return asyncio.run(
            self.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                stop=stop,
                **extras,
            )
        )

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def reset_cost(self) -> float:
        cost = self._total_cost
        self._total_cost = 0.0
        return cost

    async def close(self) -> None:
        """No-op — litellm manages its own connections."""
        pass


def create_unified_client(
    temperature: float = 0.8,
    max_tokens: int = 16384,
    timeout: float = 300.0,
) -> UnifiedLLMClient:
    """Factory function to create a UnifiedLLMClient."""
    config = UnifiedLLMClientConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return UnifiedLLMClient(config)
