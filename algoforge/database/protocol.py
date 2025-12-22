"""
RawStorage Protocol: Underlying persistence layer.

This is infrastructure - no policy decisions.
ProgramPool implementations compose over RawStorage.
"""

from dataclasses import dataclass
from typing import Protocol, Optional, runtime_checkable

from ..core import Program, EvaluationResult


@dataclass
class StorageRecord:
    """A complete record from storage."""

    id: str
    program: Program
    evaluation_result: EvaluationResult
    metadata: dict


@runtime_checkable
class RawStorage(Protocol):
    """
    Protocol for underlying persistence layer.

    This is true infrastructure - no policy decisions.
    ProgramPool implementations compose over RawStorage.
    """

    def insert(
        self,
        program: Program,
        evaluation_result: EvaluationResult,
        metadata: Optional[dict] = None
    ) -> str:
        """Insert a record, return its ID."""
        ...

    def get(self, id: str) -> Optional[StorageRecord]:
        """Get a record by ID, or None if not found."""
        ...

    def query(
        self,
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> list[StorageRecord]:
        """Query records with optional filtering and ordering."""
        ...

    def update_metadata(self, id: str, updates: dict) -> bool:
        """Update metadata for a record. Returns True if found."""
        ...

    def remove(self, id: str) -> bool:
        """Remove a record. Returns True if found."""
        ...

    def count(self, filters: Optional[dict] = None) -> int:
        """Count records matching filters."""
        ...
