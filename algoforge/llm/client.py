"""
LLMClient: Interface to language model APIs using LiteLLM

API keys via environment variables or passed directly.
"""

from dataclasses import dataclass
from typing import Optional
import random

import litellm

from ..budget import BudgetManager, ResourceType


@dataclass
class LLMConfig:
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stop: Optional[list[str]] = None


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cost: float


@dataclass
class ModelWeight:
    model: str
    weight: float  


class LLMClient:
    """
    Interface to language model APIs using LiteLLM.

    Supports ensemble of models with weighted random selection.
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
            raise ValueError("Must provide either 'model' or 'models' argument")

    def _select_model(self) -> str:
        """Select a model from ensemble based on weights."""
        selected = random.choices(self._models, weights=self._weights, k=1)[0]
        return selected.model

    def generate(
        self,
        prompt: str,
        config: Optional[LLMConfig] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        If model is not specified, selects from ensemble based on weights.

        Args:
            prompt: The prompt to send to the LLM
            config: Optional generation config (temperature, max_tokens, etc.)
            model: Optional model override (otherwise selects from ensemble)
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

        response = litellm.completion(**kwargs)

        usage = response.usage
        cost = litellm.completion_cost(completion_response=response)

        self._budget_manager.try_consume(ResourceType.LLM_TOKENS, usage.total_tokens)
        self._budget_manager.try_consume(ResourceType.LLM_COST, cost)

        content = response.choices[0].message.content

        return LLMResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model=actual_model,
            cost=cost,
        )
