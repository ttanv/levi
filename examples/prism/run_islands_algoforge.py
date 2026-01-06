#!/usr/bin/env python3
"""
Island-based AlgoForge on PRISM (ML Model Placement) Problem.

Uses the Island Model for distributed Quality-Diversity evolution:
- Multiple independent CVT-MAP-Elites archives (islands)
- Each island initialized with algorithmically-different seeds
- Ring-based migration with random elite selection
"""

from datetime import datetime

import litellm
litellm.register_model({
    "openrouter/google/gemini-2.5-flash-lite": {
        "max_tokens": 32768,
        "max_input_tokens": 1048576,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000004,
        "litellm_provider": "openrouter",
    },
    "openrouter/deepseek/deepseek-v3.2": {
        "max_tokens": 163840,
        "max_input_tokens": 163840,
        "max_output_tokens": 163840,
        "input_cost_per_token": 0.00000026,
        "output_cost_per_token": 0.00000038,
        "litellm_provider": "openrouter",
    },
    "openrouter/qwen/qwen3-coder-30b-a3b-instruct": {
        "max_tokens": 32768,
        "max_input_tokens": 160000,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.00000007,
        "output_cost_per_token": 0.00000027,
        "litellm_provider": "openrouter",
    },
    "openrouter/google/gemini-2.5-pro": {
        "max_tokens": 65536,
        "max_input_tokens": 1048576,
        "max_output_tokens": 65536,
        "input_cost_per_token": 0.00000125,
        "output_cost_per_token": 0.000010,
        "litellm_provider": "openrouter",
    },
})

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, INPUTS, score_fn
from algoforge import AlgoforgeConfig, BudgetConfig, SamplerModelPair, InitConfig, PipelineConfig, CVTConfig, MetaAdviceConfig
from algoforge.island import run_islands

# --- Model Config ---
LIGHT_MODELS = [
    "openrouter/qwen/qwen3-coder-30b-a3b-instruct",
    "openrouter/google/gemini-2.5-flash-lite",
    "openrouter/deepseek/deepseek-v3.2",
]
HEAVY_MODEL = "openrouter/deepseek/deepseek-v3.2"

# --- Island Config ---
N_ISLANDS = 3
CULLING_CHECKPOINTS = [0.5]  # Cull at 50% budget
MIGRATION_INTERVAL = 999999999  # Effectively disable migration
BUDGET_USD = 3.0

RUN_DIR = f"runs/islands_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# --- AlgoForge Config ---
config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=INPUTS,
    score_fn=score_fn,
    budget=BudgetConfig(dollars=BUDGET_USD),
    sampler_model_pairs=[
        SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[0], weight=1.0),
        SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[1], weight=1.0),
        SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[2], weight=1.0),
        SamplerModelPair(sampler="cyclic_annealing", model=HEAVY_MODEL, weight=1.0),
    ],
    cvt=CVTConfig(n_centroids=20, defer_centroids=True),
    init=InitConfig(
        n_diverse_seeds=8,  
        n_variants_per_seed=30,
        diversity_model="openrouter/deepseek/deepseek-v3.2",
        variant_model=LIGHT_MODELS[1],
    ),
    meta_advice=MetaAdviceConfig(enabled=True, interval=50, model=HEAVY_MODEL),
    pipeline=PipelineConfig(n_llm_workers=8, n_eval_processes=8, n_inspirations=1),
    output_dir=RUN_DIR,
)

# --- Run ---
if __name__ == "__main__":
    run_islands(
        config,
        n_islands=N_ISLANDS,
        culling_checkpoints=CULLING_CHECKPOINTS,
        migration_interval=MIGRATION_INTERVAL,
    )
