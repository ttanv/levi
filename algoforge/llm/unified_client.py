"""Unified LLM Client with provider abstraction and retry logic."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from .providers.base import LLMProvider, CompletionRequest, CompletionResponse
from .providers.openai_compat import OpenAICompatibleProvider
from .providers.litellm_provider import LiteLLMProvider
from .exceptions import LLMRetryExhaustedError

logger = logging.getLogger(__name__)


@dataclass
class UnifiedLLMClientConfig:
    """Configuration for the unified LLM client."""

    # Model -> endpoint URL for local models
    local_endpoints: dict[str, str] = field(default_factory=dict)

    # Model info - same format as litellm.register_model()
    # Used for cost calculation (local) and registration (cloud)
    model_info: dict[str, dict] = field(default_factory=dict)

    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

    # Default generation parameters
    default_temperature: float = 0.8
    default_max_tokens: int = 4096
    default_timeout: float = 300.0

    # Batching configuration for local models
    batch_size: int = 8
    batch_max_wait_ms: float = 50.0


class UnifiedLLMClient:
    """
    Unified LLM client that routes to appropriate provider.

    Features:
    - Automatic provider selection based on model name
    - Retry with exponential backoff
    - Cost tracking
    - Both async and sync interfaces
    """

    def __init__(self, config: UnifiedLLMClientConfig):
        self._config = config
        self._total_cost = 0.0

        # Initialize providers
        self._local_provider: Optional[OpenAICompatibleProvider] = None
        if config.local_endpoints:
            self._local_provider = OpenAICompatibleProvider(
                config.local_endpoints,
                model_info=config.model_info,
            )

        # Pass model_info to LiteLLM for registration
        self._cloud_provider = LiteLLMProvider(config.model_info if config.model_info else None)

    def _get_provider(self, model: str) -> LLMProvider:
        """Route to appropriate provider based on model."""
        if self._local_provider and self._local_provider.supports_model(model):
            return self._local_provider
        return self._cloud_provider

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
        """Async completion with retry logic."""
        provider = self._get_provider(model)

        request = CompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature if temperature is not None else self._config.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self._config.default_max_tokens,
            timeout=timeout if timeout is not None else self._config.default_timeout,
            stop=stop,
            extras=extras,
        )

        last_error: Optional[Exception] = None
        delay = self._config.retry_delay

        for attempt in range(self._config.max_retries):
            try:
                response = await provider.acompletion(request)
                self._total_cost += response.cost
                return response
            except Exception as e:
                last_error = e
                if attempt < self._config.max_retries - 1:
                    logger.warning(
                        f"[{model}] LLM call failed (attempt {attempt + 1}/{self._config.max_retries}): {e}"
                    )
                    await asyncio.sleep(delay)
                    delay *= self._config.retry_backoff

        raise LLMRetryExhaustedError(
            f"[{model}] Failed after {self._config.max_retries} retries",
            last_error,  # type: ignore
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
        """Total cost accumulated by this client."""
        return self._total_cost

    def reset_cost(self) -> float:
        """Reset cost counter and return previous value."""
        cost = self._total_cost
        self._total_cost = 0.0
        return cost

    async def close(self) -> None:
        """Close any open connections."""
        if self._local_provider:
            await self._local_provider.close()


def create_unified_client(
    local_endpoints: Optional[dict[str, str]] = None,
    cloud_registry: Optional[dict] = None,
    max_retries: int = 3,
    **kwargs,
) -> UnifiedLLMClient:
    """Factory function to create a UnifiedLLMClient.

    Args:
        local_endpoints: Map of model name -> base URL for local models
        cloud_registry: Optional dict for litellm.register_model()
        max_retries: Number of retry attempts
        **kwargs: Additional config options

    Returns:
        Configured UnifiedLLMClient instance
    """
    config = UnifiedLLMClientConfig(
        local_endpoints=local_endpoints or {},
        cloud_registry=cloud_registry,
        max_retries=max_retries,
        **kwargs,
    )
    return UnifiedLLMClient(config)
