#!/usr/bin/env python3
"""Run AlgoForge on EPLB (Expert Parallelism Load Balancer) Problem."""

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
    "openrouter/google/gemini-2.5-flash": {
        "max_tokens": 65536,
        "max_input_tokens": 1048576,
        "max_output_tokens": 65536,
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.0000025,
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
})

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, get_lazy_inputs, score_fn
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig, SamplerModelPair,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig
)

EPLB_AST_FEATURES = ['loop_nesting_max', 'subscript_count', 'math_operators', 'function_def_count']
EPLB_SCORE_KEYS = ['execution_time']

# --- Config ---
LIGHT_MODELS = [
    "openrouter/qwen/qwen3-coder-30b-a3b-instruct",
    "openrouter/google/gemini-2.5-flash-lite",
    "openrouter/deepseek/deepseek-v3.2",
]
HEAVY_MODEL = "openrouter/google/gemini-2.5-flash"

RUN_DIR = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=get_lazy_inputs(),
    score_fn=score_fn,
    budget=BudgetConfig(dollars=5.0),
    sampler_model_pairs=[
        # UCB samplers - explore underexplored behavioral regions (promotes diversity)
        SamplerModelPair(sampler="ucb", model=LIGHT_MODELS[0], weight=1.0),
        SamplerModelPair(sampler="ucb", model=LIGHT_MODELS[1], weight=1.0),
        SamplerModelPair(sampler="ucb", model=LIGHT_MODELS[2], weight=1.0),
        # Softmax samplers - exploit high-scoring solutions
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=0.7),
        SamplerModelPair(sampler="softmax", model=HEAVY_MODEL, weight=1.0, temperature=0.3),
    ],
    cvt=CVTConfig(n_centroids=50, defer_centroids=True),
    init=InitConfig(
        n_diverse_seeds=5,
        n_variants_per_seed=30,
        diversity_model=HEAVY_MODEL,
        variant_models=LIGHT_MODELS,  # Cycle through all light models
    ),
    meta_advice=MetaAdviceConfig(interval=50, model=HEAVY_MODEL),
    pipeline=PipelineConfig(n_llm_workers=8, n_eval_processes=8, n_inspirations=1),
    behavior=BehaviorConfig(ast_features=EPLB_AST_FEATURES, score_keys=EPLB_SCORE_KEYS),
    output_dir=RUN_DIR,
)

# --- Run ---
if __name__ == "__main__":
    run(config)
