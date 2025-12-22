"""
Budget management for AlgoForge.

Budget enforcement is by construction - operations check budget before proceeding.
"""

from .exceptions import BudgetExhausted
from .manager import BudgetManager, ResourceType

__all__ = [
    'BudgetExhausted',
    'BudgetManager',
    'ResourceType',
]
