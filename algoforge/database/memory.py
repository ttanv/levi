"""
InMemoryStorage: Simple in-memory implementation of RawStorage.

"""

from typing import Optional
from threading import Lock

from ..core import Program, EvaluationResult
from ..utils import generate_id
from .protocol import StorageRecord


class InMemoryStorage:
    """
    In-memory implementation of RawStorage.

    Thread-safe storage.
    """

    def __init__(self) -> None:
        self._records: dict[str, StorageRecord] = {}
        self._lock = Lock()

    def insert(
        self,
        program: Program,
        evaluation_result: EvaluationResult,
        metadata: Optional[dict] = None
    ) -> str:
        """Insert a record, return its ID."""
        record_id = generate_id()
        record = StorageRecord(
            id=record_id,
            program=program,
            evaluation_result=evaluation_result,
            metadata=metadata or {},
        )
        with self._lock:
            self._records[record_id] = record
        return record_id

    def get(self, id: str) -> Optional[StorageRecord]:
        """Get a record by ID, or None if not found."""
        with self._lock:
            return self._records.get(id)

    def query(
        self,
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> list[StorageRecord]:
        """Query records with optional filtering and ordering."""
        with self._lock:
            records = list(self._records.values())

        # Apply filters
        if filters:
            records = [r for r in records if self._matches_filters(r, filters)]

        # Apply ordering
        if order_by:
            descending = order_by.startswith('-')
            field = order_by.lstrip('-')
            records = sorted(
                records,
                key=lambda r: self._get_sort_key(r, field),
                reverse=descending
            )

        # Apply limit
        if limit:
            records = records[:limit]

        return records

    def update_metadata(self, id: str, updates: dict) -> bool:
        """Update metadata for a record. Returns True if found."""
        with self._lock:
            if id not in self._records:
                return False
            record = self._records[id]
            record.metadata.update(updates)
            return True

    def remove(self, id: str) -> bool:
        """Remove a record. Returns True if found."""
        with self._lock:
            if id in self._records:
                del self._records[id]
                return True
            return False

    def count(self, filters: Optional[dict] = None) -> int:
        """Count records matching filters."""
        if filters:
            return len(self.query(filters=filters))
        with self._lock:
            return len(self._records)

    def _matches_filters(self, record: StorageRecord, filters: dict) -> bool:
        """Check if record matches all filters."""
        for key, value in filters.items():
            if key == 'is_valid':
                if record.evaluation_result.is_valid != value:
                    return False
            elif key in record.metadata:
                if record.metadata[key] != value:
                    return False
        return True

    def _get_sort_key(self, record: StorageRecord, field: str):
        """Get sort key for a record."""
        if field == 'score':
            return record.evaluation_result.primary_score
        if field in record.evaluation_result.scores:
            return record.evaluation_result.scores[field]
        if field in record.metadata:
            return record.metadata[field]
        return 0
