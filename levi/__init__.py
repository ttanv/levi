"""
Levi: Evolutionary optimization framework for algorithms.

Simple usage::

    import levi

    result = levi.evolve_code(
        "Optimize bin packing to minimize wasted space",
        function_signature="def pack(items, bin_capacity):",
        seed_program="def pack(items, bin_capacity): ...",
        score_fn=my_scorer,
        model="openai/gpt-4o-mini",
        budget_dollars=5.0,
    )

Power users can pass any LeviConfig field as a keyword argument::

    result = levi.evolve_code(
        ...,
        paradigm_model="openai/gpt-4o",
        mutation_model="openai/gpt-4o-mini",
        budget_dollars=10.0,
        punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(enabled=True),
        pipeline=levi.PipelineConfig(n_llm_workers=8),
    )
"""

# Core types
# Behavior
from .behavior import BehaviorExtractor, FeatureVector

# Config types
from .config import (
    BehaviorConfig,
    BudgetConfig,
    CascadeConfig,
    CVTConfig,
    InitConfig,
    LeviConfig,
    LeviResult,
    MetaAdviceConfig,
    PipelineConfig,
    PromptOptConfig,
    PunctuatedEquilibriumConfig,
    SamplerModelPair,
)
from .core import EvaluationResult, MetricDict, Program

# LLM
from .llm import (
    OutputMode,
    ProgramWithScore,
    PromptBuilder,
)

# Methods
from .methods import evolve_code

# Protocols and pools
from .pool import CVTMAPElitesPool, ProgramPool, SampleResult

__version__ = "0.1.0"


__all__ = [
    # Core
    "Program",
    "EvaluationResult",
    "MetricDict",
    # Pool
    "ProgramPool",
    "SampleResult",
    "CVTMAPElitesPool",
    # LLM
    "PromptBuilder",
    "ProgramWithScore",
    "OutputMode",
    # Behavior
    "BehaviorExtractor",
    "FeatureVector",
    # Config types
    "LeviConfig",
    "LeviResult",
    "BudgetConfig",
    "SamplerModelPair",
    "CVTConfig",
    "InitConfig",
    "MetaAdviceConfig",
    "BehaviorConfig",
    "CascadeConfig",
    "PipelineConfig",
    "PunctuatedEquilibriumConfig",
    "PromptOptConfig",
    # Methods
    "evolve_code",
]
