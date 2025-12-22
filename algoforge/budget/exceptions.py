"""
Budget-related exceptions.
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import ResourceType


@dataclass
class BudgetExhausted(Exception):
    """Raised when an operation would exceed budget limits."""

    resource_type: 'ResourceType'
    remaining: Optional[float]
    requested: Optional[float] = None

    def __str__(self) -> str:
        from .manager import ResourceType

        resource_name = (
            self.resource_type.value
            if isinstance(self.resource_type, ResourceType)
            else str(self.resource_type)
        )

        if self.requested is not None:
            return (
                f"Budget exhausted: requested {self.requested} {resource_name}, "
                f"but only {self.remaining} remaining"
            )
        return f"Budget exhausted for {resource_name} (remaining: {self.remaining})"
