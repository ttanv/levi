"""Core types for AlgoForge."""

from .types import MetricDict, OutputDict, MetadataDict
from .program import Program
from .evaluation import EvaluationResult

__all__ = [
    'MetricDict',
    'OutputDict',
    'MetadataDict',
    'Program',
    'EvaluationResult',
]
