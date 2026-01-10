"""
AlgoForge: Evolutionary optimization framework for algorithms.

Example:
    import algoforge as af

    config = af.AlgoforgeConfig(
        problem_description="Optimize bin packing",
        function_signature="def solve(items) -> list[list[int]]",
        seed_program="def solve(items): ...",
        inputs=[...],
        score_fn=lambda fn, inputs: {'score': ...},
        sampler_model_pairs=[
            af.SamplerModelPair(sampler="weighted", model="gpt-4o-mini", weight=0.8),
            af.SamplerModelPair(sampler="weighted", model="gpt-4o", weight=0.2),
        ],
        budget=af.BudgetConfig(dollars=10.0),
    )

    result = af.run(config)
"""

from typing import Callable, Any, Optional

# Core types
from .core import Program, EvaluationResult, MetricDict, OutputDict

# Budget management
from .budget import BudgetManager, BudgetExhausted, ResourceType

# Protocols and pools
from .pool import ProgramPool, SampleResult, CVTMAPElitesPool
from .evaluator import Evaluator, EvaluationStage, SandboxedEvaluator
from .database import RawStorage, StorageRecord, InMemoryStorage

# LLM
from .llm import (
    LLMClient, LLMConfig, LLMResponse, ModelWeight,
    PromptBuilder, ProgramWithScore, OutputMode,
)

# Behavior
from .behavior import BehaviorExtractor, FeatureVector

# Config types
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
    CascadeConfig,
    PipelineConfig,
)

# Methods
from .methods import run

__version__ = "0.1.0"


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
    # Config types
    'AlgoforgeConfig',
    'AlgoforgeResult',
    'BudgetConfig',
    'SamplerModelPair',
    'CVTConfig',
    'InitConfig',
    'MetaAdviceConfig',
    'BehaviorConfig',
    'CheckpointConfig',
    'CascadeConfig',
    'PipelineConfig',
    # Methods
    'run',
]
