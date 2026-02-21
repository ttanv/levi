"""Unified LLM Client with provider abstraction and retry logic."""

import asyncio
import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse
from typing import Optional

from .providers.base import LLMProvider, CompletionRequest, CompletionResponse
from .providers.openai_compat import OpenAICompatibleProvider
from .providers.litellm_provider import LiteLLMProvider
from .exceptions import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMRetryExhaustedError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResolver:
    """Resolve model -> provider route and validate local endpoint config."""

    local_endpoints: dict[str, str] = field(default_factory=dict)
    known_models: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        normalized: dict[str, str] = {}
        for model, endpoint in self.local_endpoints.items():
            model_name = model.strip()
            base_url = endpoint.strip()
            if not model_name:
                raise LLMConfigurationError("local_endpoints contains an empty model name")
            if not base_url:
                raise LLMConfigurationError(f"local_endpoints[{model_name!r}] has an empty URL")
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise LLMConfigurationError(
                    f"Invalid endpoint URL for model {model_name!r}: {base_url!r}"
                )
            normalized[model_name] = base_url.rstrip("/")

        self.local_endpoints = normalized

        if self.known_models:
            unused = sorted(m for m in self.local_endpoints if m not in self.known_models)
            if unused:
                logger.warning(
                    "Unused local_endpoints entries: %s",
                    ", ".join(unused),
                )

    def is_local(self, model: str) -> bool:
        if not model or not model.strip():
            raise LLMConfigurationError("Model name must be a non-empty string")
        return model in self.local_endpoints


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

    # Models referenced by runtime config; used for startup validation warnings
    known_models: set[str] = field(default_factory=set)


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
        self._resolver = ModelResolver(config.local_endpoints, known_models=config.known_models)

        if config.max_retries < 1:
            raise LLMConfigurationError("max_retries must be >= 1")
        if config.retry_delay <= 0:
            raise LLMConfigurationError("retry_delay must be > 0")
        if config.retry_backoff < 1:
            raise LLMConfigurationError("retry_backoff must be >= 1")

        # Initialize providers
        self._local_provider: Optional[OpenAICompatibleProvider] = None
        if self._resolver.local_endpoints:
            self._local_provider = OpenAICompatibleProvider(
                self._resolver.local_endpoints,
                model_info=config.model_info,
            )

        # Pass model_info to LiteLLM for registration
        self._cloud_provider = LiteLLMProvider(config.model_info if config.model_info else None)

    def _get_provider(self, model: str) -> LLMProvider:
        """Route to appropriate provider based on model."""
        if self._resolver.is_local(model):
            if not self._local_provider:
                raise LLMConfigurationError(
                    f"Model '{model}' resolved as local but local provider is not configured"
                )
            return self._local_provider
        return self._cloud_provider

    @staticmethod
    def _should_retry(error: Exception) -> bool:
        return isinstance(error, (LLMConnectionError, LLMTimeoutError, LLMRateLimitError))

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
                if not self._should_retry(e):
                    raise
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
    model_info: Optional[dict[str, dict]] = None,
    cloud_registry: Optional[dict[str, dict]] = None,
    known_models: Optional[set[str]] = None,
    max_retries: int = 3,
    **kwargs,
) -> UnifiedLLMClient:
    """Factory function to create a UnifiedLLMClient.

    Args:
        local_endpoints: Map of model name -> base URL for local models
        model_info: Optional model metadata in LiteLLM register_model format
        cloud_registry: Deprecated alias for model_info
        known_models: Optional set of runtime model names for validation warnings
        max_retries: Number of retry attempts
        **kwargs: Additional config options

    Returns:
        Configured UnifiedLLMClient instance
    """
    resolved_model_info = model_info or cloud_registry or {}

    resolved_known_models = set(known_models) if known_models is not None else set()

    config = UnifiedLLMClientConfig(
        local_endpoints=local_endpoints or {},
        model_info=resolved_model_info,
        known_models=resolved_known_models,
        max_retries=max_retries,
        **kwargs,
    )
    return UnifiedLLMClient(config)
