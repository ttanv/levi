"""
Program: The fundamental unit of evolution.

A Program represents a candidate solution being optimized.
Programs are immutable once created - mutations produce new Program instances.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import uuid

from .types import MetadataDict


def _generate_id() -> str:
    """Generate a unique program ID."""
    return str(uuid.uuid4())


@dataclass(frozen=True)
class Program:
    """
    The fundamental unit of evolution.

    A Program represents a candidate solution being optimized.
    Programs are immutable once created - mutations produce new Program instances.

    Attributes:
        code: The actual source code (string)
        id: Unique identifier
        parents: Tuple of parent program IDs (empty for seed programs)
        metadata: Arbitrary key-value store for method-specific data
        created_at: Timestamp of creation
    """

    code: str
    id: str = field(default_factory=_generate_id)
    parents: tuple[str, ...] = field(default_factory=tuple)
    metadata: MetadataDict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def with_mutation(
        self,
        new_code: str,
        additional_metadata: Optional[MetadataDict] = None
    ) -> 'Program':
        """
        Create a new Program by mutating this one.

        Args:
            new_code: The mutated source code
            additional_metadata: Extra metadata to merge with existing

        Returns:
            A new Program with this program as its single parent
        """
        merged_metadata = {**self.metadata, **(additional_metadata or {})}
        return Program(
            code=new_code,
            parents=(self.id,),
            metadata=merged_metadata,
        )

    def with_crossover(
        self,
        other: 'Program',
        new_code: str,
        additional_metadata: Optional[MetadataDict] = None
    ) -> 'Program':
        """
        Create a new Program by crossing over with another.

        Args:
            other: The other parent program
            new_code: The crossover result code
            additional_metadata: Extra metadata to merge

        Returns:
            A new Program with both programs as parents
        """
        merged_metadata = {**self.metadata, **(additional_metadata or {})}
        return Program(
            code=new_code,
            parents=(self.id, other.id),
            metadata=merged_metadata,
        )

    @property
    def is_seed(self) -> bool:
        """Returns True if this is a seed program (no parents)."""
        return len(self.parents) == 0
