"""LiteLLM provider for cloud APIs (OpenRouter, Anthropic, OpenAI, etc.)."""

import asyncio
from typing import Optional

import litellm
litellm.suppress_debug_info = True

from .base import LLMProvider, CompletionRequest, CompletionResponse


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

        response = await litellm.acompletion(**kwargs)

        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            # Cost calculation may fail for some models
            cost = 0.0

        usage = response.usage
        return CompletionResponse(
            content=response.choices[0].message.content,
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

        response = litellm.completion(**kwargs)

        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        usage = response.usage
        return CompletionResponse(
            content=response.choices[0].message.content,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model=request.model,
            cost=cost,
            raw_response=response,
        )
