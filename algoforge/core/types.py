"""
Type aliases used throughout AlgoForge.
"""

from typing import Any

# Score/metric dictionary: maps metric names to scalar values
MetricDict = dict[str, float]

# Output dictionary: maps input identifiers to program outputs
OutputDict = dict[str, Any]

# Metadata dictionary: arbitrary key-value store
MetadataDict = dict[str, Any]
