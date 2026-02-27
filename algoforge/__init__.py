"""
AlgoForge: Evolutionary optimization framework for algorithms.

Simple usage::

    import algoforge as af

    result = af.evolve_code(
        "Optimize bin packing to minimize wasted space",
        function_signature="def pack(items, bin_capacity):",
        seed_program="def pack(items, bin_capacity): ...",
        score_fn=my_scorer,
        model="openai/gpt-4o-mini",
        budget_dollars=5.0,
    )

Power users can pass any AlgoforgeConfig field as a keyword argument::

    result = af.evolve_code(
        ...,
        paradigm_model="openai/gpt-4o",
        mutation_model="openai/gpt-4o-mini",
        budget_dollars=10.0,
        punctuated_equilibrium=af.PunctuatedEquilibriumConfig(enabled=True),
        pipeline=af.PipelineConfig(n_llm_workers=8),
    )
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
    PromptOptConfig,
)

# Methods
from .methods import evolve_code

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
    'PromptOptConfig',
    # Methods
    'evolve_code',
]
