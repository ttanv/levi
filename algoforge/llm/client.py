"""
LLMClient: Interface to language model APIs using LiteLLM.

Supports:
- Any LiteLLM-compatible provider (Anthropic, OpenAI, Azure, OpenRouter, Google, etc.)
- Ensemble of models with weighted selection

Model string formats:
- Anthropic: "claude-sonnet-4-20250514", "claude-3-opus-20240229"
- OpenAI: "gpt-4", "gpt-4o", "gpt-3.5-turbo"
- Azure: "azure/your-deployment-name"
- OpenRouter: "openrouter/anthropic/claude-3-opus"
- Google: "gemini/gemini-pro"
- Together: "together_ai/meta-llama/Llama-3-70b"

API keys via environment variables or passed directly.
"""

from dataclasses import dataclass
from typing import Optional
import random

import litellm

from ..budget import BudgetManager, ResourceType


@dataclass
class LLMConfig:
    """Configuration for LLM generation."""
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stop: Optional[list[str]] = None


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cost: float
    is_structured: bool = False  # Indicates if this is structured output
    parsed_json: Optional[dict] = None  # Parsed JSON for structured outputs


@dataclass
class ModelWeight:
    """A model with its selection weight."""
    model: str
    weight: float  # e.g., 0.8 for 80% of calls


class LLMClient:
    """
    Interface to language model APIs using LiteLLM.

    Supports ensemble of models with weighted random selection.

    Example:
        # Single model
        client = LLMClient(budget, model="gpt-4o")

        # Ensemble: 80% fast model, 20% strong model
        client = LLMClient(budget, models=[
            ModelWeight("gpt-4o-mini", 0.8),
            ModelWeight("gpt-4o", 0.2),
        ])
    """

    def __init__(
        self,
        budget_manager: BudgetManager,
        model: Optional[str] = None,
        models: Optional[list[ModelWeight]] = None,
        default_config: Optional[LLMConfig] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        """
        Args:
            budget_manager: Budget manager for tracking LLM costs
            model: Single model identifier (use this OR models, not both)
            models: List of ModelWeight for ensemble selection
            default_config: Default generation configuration
            api_key: API key (or use environment variable)
            api_base: API base URL for Azure/custom endpoints
        """
        self._budget_manager = budget_manager
        self.default_config = default_config or LLMConfig()
        self._api_key = api_key
        self._api_base = api_base

        if models:
            self._models = models
            self._weights = [m.weight for m in models]
        elif model:
            self._models = [ModelWeight(model, 1.0)]
            self._weights = [1.0]
        else:
            self._models = [ModelWeight("gpt-4o-mini", 1.0)]
            self._weights = [1.0]

    def _select_model(self) -> str:
        """Select a model from ensemble based on weights."""
        selected = random.choices(self._models, weights=self._weights, k=1)[0]
        return selected.model

    def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        model: Optional[str] = None,
        response_format: Optional[dict] = None
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        If model is not specified, selects from ensemble based on weights.

        Args:
            prompt: The prompt to send to the LLM
            config: Optional generation config (temperature, max_tokens, etc.)
            model: Optional model override (otherwise selects from ensemble)
            response_format: Optional JSON schema for structured outputs.
                Format: {"type": "json_schema", "json_schema": {...}}
                See: https://docs.litellm.ai/docs/completion/json_mode
        """
        self._budget_manager.check_budget()

        cfg = config or self.default_config
        actual_model = model or self._select_model()

        kwargs = {
            "model": actual_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
        }
        if cfg.stop:
            kwargs["stop"] = cfg.stop
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if response_format:
            kwargs["response_format"] = response_format
            # Enable response-healing for OpenRouter models with structured outputs
            if actual_model.startswith("openrouter/"):
                extra_body = kwargs.get("extra_body", {})
                extra_body["plugins"] = [{"id": "response-healing"}]
                kwargs["extra_body"] = extra_body

        response = litellm.completion(**kwargs)

        usage = response.usage
        cost = litellm.completion_cost(completion_response=response)

        self._budget_manager.try_consume(ResourceType.LLM_TOKENS, usage.total_tokens)
        self._budget_manager.try_consume(ResourceType.LLM_COST, cost)

        content = response.choices[0].message.content

        # Parse JSON if structured output was requested
        parsed_json = None
        is_structured = response_format is not None
        if is_structured:
            try:
                import json
                parsed_json = json.loads(content)
            except json.JSONDecodeError:
                # This shouldn't happen with strict schemas, but handle gracefully
                is_structured = False

        return LLMResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model=actual_model,
            cost=cost,
            is_structured=is_structured,
            parsed_json=parsed_json,
        )

    def generate_batch(
        self,
        prompts: list[str],
        config: Optional[LLMConfig] = None
    ) -> list[LLMResponse]:
        """Generate completions for multiple prompts."""
        return [self.generate(p, config) for p in prompts]
