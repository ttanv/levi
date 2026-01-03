"""
AlgoForge: Evolutionary optimization framework for algorithms.

Example:
    import algoforge as af

    af.configure(models=[
        af.ModelWeight('gpt-4o-mini', 0.8),
        af.ModelWeight('gpt-4o', 0.2),
    ])

    best = af.alphaevolve(
        score_functions={'score': lambda out, inp, t: -len(out)},
        inputs=[...],
        seed_program='def solve(x): ...',
        problem_description='Optimize bin packing',
        function_signature='def solve(items) -> list[list[int]]',
        budget_evaluations=100,
    )
"""

from typing import Callable, Any, Optional

# Core types
from .core import Program, EvaluationResult, MetricDict, OutputDict

# Budget management
from .budget import BudgetManager, BudgetExhausted, ResourceType

# Protocols and pools
from .pool import ProgramPool, SampleResult, SimplePool, CVTMAPElitesPool
from .evaluator import Evaluator, EvaluationStage, SandboxedEvaluator
from .database import RawStorage, StorageRecord, InMemoryStorage

# LLM
from .llm import (
    LLMClient, LLMConfig, LLMResponse, ModelWeight,
    PromptBuilder, ProgramWithScore, OutputMode,
)

# Behavior
from .behavior import BehaviorExtractor, FeatureVector

# Configuration
from ._config import configure, get_config, AlgoForgeConfig

# New config types
from .config import (
    AlgoforgeConfig,
    AlgoforgeResult,
    BudgetConfig,
    SamplerModelPair,
    CVTConfig,
    InitConfig,
    MetaAdviceConfig,
    BehaviorConfig,
    CheckpointConfig,
    PipelineConfig,
)

# Methods
from .methods import alphaevolve, run

__version__ = "0.1.0"

# Stubs for unimplemented methods

def funsearch(
    evaluate: Callable[[Any, Any], float],
    instances: list,
    budget_rollouts: Optional[int] = None,
    budget_dollars: Optional[float] = None,
    budget_seconds: Optional[float] = None,
    **kwargs
) -> Program:
    """Run FunSearch algorithm."""
    raise NotImplementedError("FunSearch method not yet implemented")


def gepa(
    evaluate: Callable[[Any, Any], float],
    instances: list,
    budget_rollouts: Optional[int] = None,
    budget_dollars: Optional[float] = None,
    budget_seconds: Optional[float] = None,
    **kwargs
) -> Program:
    """Run GEPA algorithm with reflective evolution."""
    raise NotImplementedError("GEPA method not yet implemented")


def discover(
    evaluate: Callable[[Any, Any], float],
    instances: list,
    **kwargs
) -> Program:
    """Auto-select best method and run discovery."""
    raise NotImplementedError("Discovery method not yet implemented")


__all__ = [
    # Core
    'Program',
    'EvaluationResult',
    'MetricDict',
    'OutputDict',
    # Budget
    'BudgetManager',
    'BudgetExhausted',
    'ResourceType',
    # Pool
    'ProgramPool',
    'SampleResult',
    'SimplePool',
    'CVTMAPElitesPool',
    # Evaluator
    'Evaluator',
    'EvaluationStage',
    'SandboxedEvaluator',
    # Database
    'RawStorage',
    'StorageRecord',
    'InMemoryStorage',
    # LLM
    'LLMClient',
    'LLMConfig',
    'LLMResponse',
    'ModelWeight',
    'PromptBuilder',
    'ProgramWithScore',
    'OutputMode',
    # Behavior
    'BehaviorExtractor',
    'FeatureVector',
    # Legacy config
    'configure',
    'get_config',
    'AlgoForgeConfig',
    # New config types
    'AlgoforgeConfig',
    'AlgoforgeResult',
    'BudgetConfig',
    'SamplerModelPair',
    'CVTConfig',
    'InitConfig',
    'MetaAdviceConfig',
    'BehaviorConfig',
    'CheckpointConfig',
    'PipelineConfig',
    # Methods
    'run',
    'alphaevolve',
    'funsearch',
    'gepa',
    'discover',
]
