"""Minimal client abstractions for text-generation backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_TIMEOUT: float = 300.0
ClientInput = str | list[dict[str, Any]]


@dataclass(slots=True)
class ClientResult:
    """Normalized text-generation result."""

    text: str
    cost: float = 0.0


class BaseClient(ABC):
    """Base class for all generation backends used by Levi."""

    def __init__(
        self,
        model: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        input_cost_per_token: Optional[float] = None,
        output_cost_per_token: Optional[float] = None,
        cache_read_input_token_cost: Optional[float] = None,
        **defaults: Any,
    ):
        self.model = model
        self.timeout = timeout
        self.input_cost_per_token = input_cost_per_token
        self.output_cost_per_token = output_cost_per_token
        self.cache_read_input_token_cost = cache_read_input_token_cost
        self.defaults = defaults

    @abstractmethod
    async def acompletion(self, prompt: ClientInput, **kwargs: Any) -> ClientResult:
        raise NotImplementedError

    async def close(self) -> None:
        """Release any backend resources held by the client."""
        return None


ClientSpec = str | BaseClient
