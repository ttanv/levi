"""
Optimization methods for AlgoForge.

Methods compose primitives into complete optimization algorithms.
"""

from .alphaevolve import alphaevolve
from .algoforge import run

__all__ = ['alphaevolve', 'run']
