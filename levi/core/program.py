"""
Program: The fundamental unit of evolution.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from .types import MetadataDict


def _generate_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class Program:
    """
    The fundamental unit of evolution.
    """

    content: str
    id: str = field(default_factory=_generate_id)
    metadata: MetadataDict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
