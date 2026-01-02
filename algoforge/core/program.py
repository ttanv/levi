"""
Program: The fundamental unit of evolution.
"""

from dataclasses import dataclass, field
from datetime import datetime
import uuid

from .types import MetadataDict


def _generate_id() -> str:
    """Generate a unique program ID."""
    return str(uuid.uuid4())


@dataclass(frozen=True)
class Program:
    """
    The fundamental unit of evolution.
    Programs are immutable once created.

    Attributes:
        code: The actual code/string
        id: Unique identifier
        parents: Tuple of parent program IDs
        metadata: Arbitrary key-value store for method-specific data
        created_at: Timestamp of creation
    """

    code: str
    id: str = field(default_factory=_generate_id)
    parents: tuple[str, ...] = field(default_factory=tuple)
    metadata: MetadataDict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def is_seed(self) -> bool:
        """Returns True if this is a seed program (no parents)."""
        return len(self.parents) == 0
