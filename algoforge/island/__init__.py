"""
Island Model for Distributed Quality-Diversity Evolution.

Implements a ring-based island topology with independent CVT archives
per island and random elite migration for maintaining algorithmic diversity.
"""

from .coordinator import Island, IslandCoordinator
from .diversifier import IslandDiversifier
from .runner import run_islands, run_islands_async, IslandPipelineRunner

__all__ = [
    "Island",
    "IslandCoordinator",
    "IslandDiversifier",
    "IslandPipelineRunner",
    "run_islands",
    "run_islands_async",
]
