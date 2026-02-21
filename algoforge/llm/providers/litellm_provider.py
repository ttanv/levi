"""LiteLLM provider for cloud APIs (OpenRouter, Anthropic, OpenAI, etc.)."""

import asyncio
from typing import Optional

import litellm
litellm.suppress_debug_info = True

from .base import LLMProvider, CompletionRequest, CompletionResponse
from ..exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)


def _wrap_litellm_error(model: str, error: Exception) -> Exception:
    """Map provider exceptions into internal LLM exception types."""
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


class LiteLLMProvider(LLMProvider):
    """Provider using LiteLLM for cloud APIs."""

    def __init__(self, model_registry: Optional[dict] = None):
        """
        Args:
            model_registry: Optional dict to pass to litellm.register_model()
                for custom model configurations.
        """
        if model_registry:
            litellm.register_model(model_registry)

    def supports_model(self, model: str) -> bool:
        # Cloud models typically have provider prefixes or known patterns
        # This is a fallback provider, so it claims to support everything
        # that isn't explicitly handled by a local provider
        return True

    async def acompletion(self, request: CompletionRequest) -> CompletionResponse:
        kwargs = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "timeout": request.timeout,
        }
        if request.stop:
            kwargs["stop"] = request.stop

        # Add any extra parameters (api_key, etc.)
        kwargs.update(request.extras)

        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as e:
            raise _wrap_litellm_error(request.model, e) from e

        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            # Cost calculation may fail for some models
            cost = 0.0

        usage = getattr(response, "usage", None)
        if usage is None:
            raise LLMResponseError(f"[{request.model}] Missing usage field in response")

        content = None
        try:
            content = response.choices[0].message.content
        except Exception as e:
            raise LLMResponseError(f"[{request.model}] Invalid completion schema") from e

        return CompletionResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model=request.model,
            cost=cost,
            raw_response=response,
        )

    def completion(self, request: CompletionRequest) -> CompletionResponse:
        kwargs = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "timeout": request.timeout,
        }
        if request.stop:
            kwargs["stop"] = request.stop

        kwargs.update(request.extras)

        try:
            response = litellm.completion(**kwargs)
        except Exception as e:
            raise _wrap_litellm_error(request.model, e) from e

        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        usage = getattr(response, "usage", None)
        if usage is None:
            raise LLMResponseError(f"[{request.model}] Missing usage field in response")

        content = None
        try:
            content = response.choices[0].message.content
        except Exception as e:
            raise LLMResponseError(f"[{request.model}] Invalid completion schema") from e

        return CompletionResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model=request.model,
            cost=cost,
            raw_response=response,
        )
