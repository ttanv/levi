"""LLM integration for Levi."""

from .context import clear_llm_client, get_llm_client, set_llm_client
from .exceptions import LLMError
from .prompts import OutputMode, ProgramWithScore, PromptBuilder
from .unified_client import (
    CompletionResponse,
    UnifiedLLMClient,
    UnifiedLLMClientConfig,
    create_unified_client,
)

__all__ = [
    # Prompt utilities
    "PromptBuilder",
    "ProgramWithScore",
    "OutputMode",
    # Unified client
    "UnifiedLLMClient",
    "UnifiedLLMClientConfig",
    "create_unified_client",
    "CompletionResponse",
    # Context management
    "get_llm_client",
    "set_llm_client",
    "clear_llm_client",
    # Exceptions
    "LLMError",
]
