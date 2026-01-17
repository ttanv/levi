"""Context management for LLM client dependency injection."""

from contextvars import ContextVar
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .unified_client import UnifiedLLMClient

_llm_client: ContextVar[Optional["UnifiedLLMClient"]] = ContextVar(
    "llm_client", default=None
)


def get_llm_client() -> "UnifiedLLMClient":
    """Get the current LLM client from context.

    Raises:
        RuntimeError: If no client has been set.
    """
    client = _llm_client.get()
    if client is None:
        raise RuntimeError(
            "LLM client not initialized. Call set_llm_client() first."
        )
    return client


def set_llm_client(client: "UnifiedLLMClient") -> None:
    """Set the LLM client in context."""
    _llm_client.set(client)


def clear_llm_client() -> None:
    """Clear the LLM client from context."""
    _llm_client.set(None)
