"""Storage layer for AlgoForge."""

from .protocol import RawStorage, StorageRecord
from .memory import InMemoryStorage

__all__ = [
    'RawStorage',
    'StorageRecord',
    'InMemoryStorage',
]
