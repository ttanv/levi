"""LLM integration for AlgoForge."""

from .prompts import PromptBuilder, ProgramWithScore, OutputMode
from .unified_client import (
    UnifiedLLMClient,
    UnifiedLLMClientConfig,
    create_unified_client,
    CompletionResponse,
)
from .context import get_llm_client, set_llm_client, clear_llm_client
from .exceptions import LLMError

__all__ = [
    # Prompt utilities
    'PromptBuilder',
    'ProgramWithScore',
    'OutputMode',
    # Unified client
    'UnifiedLLMClient',
    'UnifiedLLMClientConfig',
    'create_unified_client',
    'CompletionResponse',
    # Context management
    'get_llm_client',
    'set_llm_client',
    'clear_llm_client',
    # Exceptions
    'LLMError',
]
