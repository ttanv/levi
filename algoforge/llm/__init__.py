"""LLM integration for AlgoForge."""

from .prompts import PromptBuilder, ProgramWithScore, OutputMode
from .unified_client import UnifiedLLMClient, UnifiedLLMClientConfig, create_unified_client
from .context import get_llm_client, set_llm_client, clear_llm_client
from .exceptions import LLMError, LLMRetryExhaustedError
from .providers import (
    LLMProvider,
    CompletionRequest,
    CompletionResponse,
    OpenAICompatibleProvider,
    LiteLLMProvider,
)

__all__ = [
    # Prompt utilities
    'PromptBuilder',
    'ProgramWithScore',
    'OutputMode',
    # New unified client
    'UnifiedLLMClient',
    'UnifiedLLMClientConfig',
    'create_unified_client',
    # Context management
    'get_llm_client',
    'set_llm_client',
    'clear_llm_client',
    # Exceptions
    'LLMError',
    'LLMRetryExhaustedError',
    # Provider abstractions
    'LLMProvider',
    'CompletionRequest',
    'CompletionResponse',
    'OpenAICompatibleProvider',
    'LiteLLMProvider',
]
