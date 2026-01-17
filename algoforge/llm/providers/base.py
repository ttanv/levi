"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class CompletionRequest:
    """Request for an LLM completion."""

    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.8
    max_tokens: int = 4096
    stop: Optional[list[str]] = None
    timeout: float = 300.0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResponse:
    """Response from an LLM completion."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cost: float
    raw_response: Optional[Any] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def acompletion(self, request: CompletionRequest) -> CompletionResponse:
        """Async completion call."""
        pass

    @abstractmethod
    def completion(self, request: CompletionRequest) -> CompletionResponse:
        """Sync completion call."""
        pass

    @abstractmethod
    def supports_model(self, model: str) -> bool:
        """Check if this provider handles the given model."""
        pass
