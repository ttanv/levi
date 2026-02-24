#!/usr/bin/env python3
"""Run AlgoForge with Prompt Optimization for Can't Be Late Scheduling."""

import sys
from datetime import datetime
from pathlib import Path

# Add algoforge to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, INPUTS, score_fn, DIVERSITY_SEED_PROMPT
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig, SamplerModelPair,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig,
    PunctuatedEquilibriumConfig, PromptOptConfig,
)

# Behavioral dimensions for cant_be_late
CANT_BE_LATE_AST_FEATURES = ['cyclomatic_complexity', 'comparison_count', 'math_operators', 'branch_count']
CANT_BE_LATE_SCORE_KEYS = ['tight_deadline_score', 'loose_deadline_score']

# Models
LIGHT_MODELS = [
    "openrouter/xiaomi/mimo-v2-flash",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
]
PARADIGM_SHIFT_MODEL = "openrouter/google/gemini-3-flash-preview"

LOCAL_ENDPOINTS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8001/v1",
}

MODEL_INFO = {
    "xiaomi/mimo-v2-flash": {
        "max_tokens": 16384,
        "max_input_tokens": 262144,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000009,
        "output_cost_per_token": 0.00000029,
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000004,
    },
    "google/gemini-3-flash-preview": {
        "max_tokens": 65536,
        "max_input_tokens": 1048576,
        "max_output_tokens": 65536,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.000003,
    },
}

run_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_po"

config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=INPUTS,
    score_fn=score_fn,
    budget=BudgetConfig(dollars=5.00),
    sampler_model_pairs=[
        # MiMo-V2-Flash (OpenRouter)
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=0.3),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=0.7),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=1.0),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=1.2),
        # Qwen 30B (Local TPU)
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=0.3),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=0.7),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=1.0),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=1.2),
        # Gemini Flash (OpenRouter) - paradigm shift model
        SamplerModelPair(sampler="softmax", model=PARADIGM_SHIFT_MODEL, weight=1.0, temperature=0.3),
    ],
    cvt=CVTConfig(n_centroids=50, defer_centroids=True),
    init=InitConfig(
        enabled=True,
        n_diverse_seeds=5,
        n_variants_per_seed=20,
        diversity_model=PARADIGM_SHIFT_MODEL,
        variant_models=LIGHT_MODELS,
        temperature=0.8,
        diversity_prompt=DIVERSITY_SEED_PROMPT,
    ),
    meta_advice=MetaAdviceConfig(enabled=True, interval=50, model=PARADIGM_SHIFT_MODEL),
    pipeline=PipelineConfig(n_llm_workers=12, n_eval_processes=12, n_inspirations=1, output_mode="full", eval_timeout=300),
    behavior=BehaviorConfig(ast_features=CANT_BE_LATE_AST_FEATURES, score_keys=CANT_BE_LATE_SCORE_KEYS, init_noise=0.0),
    punctuated_equilibrium=PunctuatedEquilibriumConfig(
        enabled=True,
        interval=5,
        n_clusters=3,
        n_variants=3,
        heavy_models=[PARADIGM_SHIFT_MODEL],
        variant_models=LIGHT_MODELS,
        behavior_noise=0.0,
        temperature=0.7,
        reasoning_effort="low",
    ),
    prompt_opt=PromptOptConfig(enabled=True),
    output_dir=run_dir,
    local_endpoints=LOCAL_ENDPOINTS,
    model_info=MODEL_INFO,
)

if __name__ == "__main__":
    run(config)
