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
        paradigm_models="openai/gpt-4o",
        mutation_models="openai/gpt-4o-mini",
        budget=af.BudgetConfig(dollars=10.0),
    )

    result = af.run(config)
"""

# Core types
from .core import Program, EvaluationResult, MetricDict

# Protocols and pools
from .pool import ProgramPool, SampleResult, CVTMAPElitesPool

# LLM
from .llm import (
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
    CascadeConfig,
    PipelineConfig,
    PunctuatedEquilibriumConfig,
)

# Methods
from .methods import run

__version__ = "0.1.0"


__all__ = [
    # Core
    'Program',
    'EvaluationResult',
    'MetricDict',
    # Pool
    'ProgramPool',
    'SampleResult',
    'CVTMAPElitesPool',
    # LLM
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
    'CascadeConfig',
    'PipelineConfig',
    'PunctuatedEquilibriumConfig',
    # Methods
    'run',
]
