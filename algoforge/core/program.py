"""
Program: The fundamental unit of evolution.
"""

from dataclasses import dataclass, field
from datetime import datetime
import uuid

from .types import MetadataDict


def _generate_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class Program:
    """
    The fundamental unit of evolution.
    """

    code: str
    id: str = field(default_factory=_generate_id)
    parents: tuple[str, ...] = field(default_factory=tuple)
    metadata: MetadataDict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
