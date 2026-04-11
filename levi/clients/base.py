"""Minimal client abstractions for text-generation backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

ClientInput = str | list[dict[str, Any]]


@dataclass(slots=True)
class ClientResult:
    """Normalized text-generation result."""

    text: str
    cost: float = 0.0


class BaseClient(ABC):
    """Base class for all generation backends used by Levi."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def acompletion(self, prompt: ClientInput, **kwargs: Any) -> ClientResult:
        raise NotImplementedError

    async def close(self) -> None:
        """Release any backend resources held by the client."""
        return None


ClientSpec = str | BaseClient


def client_name(spec: ClientSpec) -> str:
    """Return the model identifier associated with a client spec."""
    if isinstance(spec, BaseClient):
        return spec.model
    return spec


def short_client_name(spec: ClientSpec) -> str:
    """Return the trailing segment of a model identifier for logs."""
    return client_name(spec).split("/")[-1]
