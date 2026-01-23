#!/usr/bin/env python3
"""Run AlgoForge on EPLB (Expert Parallelism Load Balancer) Problem."""

from datetime import datetime

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, get_lazy_inputs, score_fn
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig, SamplerModelPair,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig
)
from algoforge.config.models import PunctuatedEquilibriumConfig, LLMProviderConfig

# Behavioral dimensions: 3 code structure + 4 performance profile = 7 total
EPLB_AST_FEATURES = ['loop_nesting_max', 'cyclomatic_complexity', 'math_operators']
EPLB_SCORE_KEYS = ['execution_time', 'workload_main', 'workload_8', 'workload_9']

# --- Config ---
# Local TPU models (raw model names as vLLM expects them)
# Use load balancer on port 8001 (run load_balancer.py first)
LIGHT_MODELS = [
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
]
HEAVY_MODEL = "openrouter/xiaomi/mimo-v2-flash"

# Model -> API base URL mapping for local TPU endpoints
LOCAL_ENDPOINTS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8001/v1",
}

# Model info for token cost tracking (same format as litellm.register_model)
MODEL_INFO = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000004,
    },
    "xiaomi/mimo-v2-flash": {
        "max_tokens": 32768,
        "max_input_tokens": 262144,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.00000009,   # $0.09/M input
        "output_cost_per_token": 0.00000029,  # $0.29/M output
    },
    "google/gemini-3-flash-preview": {
        "max_tokens": 32768,
        "max_input_tokens": 1048576,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0000005,    # $0.50/M input
        "output_cost_per_token": 0.000003,    # $3/M output
    },
}

RUN_DIR = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=get_lazy_inputs(),
    score_fn=score_fn,
    budget=BudgetConfig(dollars=3),
    sampler_model_pairs=[
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=0.3),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=0.7),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=1.0),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=1.2),
    ],
    cvt=CVTConfig(n_centroids=40, defer_centroids=True, predefined_centroids_file="/home/ttanveer/algoforge/examples/eplb/centroids.json"),
    init=InitConfig(
        enabled=True,
        n_diverse_seeds=3,
        n_variants_per_seed=5,
        diversity_model="openrouter/google/gemini-3-flash-preview",
        variant_models=LIGHT_MODELS,
        temperature=0.8,
    ),
    meta_advice=MetaAdviceConfig(enabled=False, interval=50, model=HEAVY_MODEL),
    pipeline=PipelineConfig(n_llm_workers=8, n_eval_processes=8, n_inspirations=1, output_mode="diff"),
    behavior=BehaviorConfig(ast_features=EPLB_AST_FEATURES, score_keys=EPLB_SCORE_KEYS),
    punctuated_equilibrium=PunctuatedEquilibriumConfig(
        enabled=True,
        interval=5,
        n_clusters=3,
        n_variants=3,
        heavy_model=HEAVY_MODEL,
        variant_models=LIGHT_MODELS,
        behavior_noise=0.3,
        temperature=0.7,
        reasoning_effort="disabled",
    ),
    output_dir=RUN_DIR,
    llm=LLMProviderConfig(
        local_endpoints=LOCAL_ENDPOINTS,
        model_info=MODEL_INFO,
    ),
)

# --- Run ---
if __name__ == "__main__":
    run(config)
