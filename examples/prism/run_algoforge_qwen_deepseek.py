#!/usr/bin/env python3
"""Run AlgoForge on PRISM (GPU Model Placement) Problem - Qwen light + DeepSeek heavy."""

from datetime import datetime

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, get_lazy_inputs, score_fn
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig, SamplerModelPair,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig
)
from algoforge.config.models import PunctuatedEquilibriumConfig, LLMProviderConfig

# Behavioral dimensions: AST features that distinguish algorithmic approaches
# - loop_nesting_max: O(n) vs O(n²) vs O(n³) complexity
# - branch_count: Simple vs heuristic-heavy logic
# - comparison_count: Linear vs binary search intensity
# - subscript_count: List-append vs index-manipulation style
PRISM_AST_FEATURES = ['loop_nesting_max', 'branch_count', 'comparison_count', 'subscript_count']
PRISM_SCORE_KEYS = []  # Use only AST features for behavioral diversity

# --- Config ---
# Local TPU models (raw model names as vLLM expects them)
# Use load balancer on port 8001 (run load_balancer.py first)
LIGHT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
HEAVY_MODEL = "openrouter/deepseek/deepseek-v3.2"

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
    "deepseek/deepseek-v3.2": {
        "max_tokens": 16384,
        "max_input_tokens": 163840,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000025,   # $0.25/M input
        "output_cost_per_token": 0.00000038,  # $0.38/M output
    },
}

RUN_DIR = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=get_lazy_inputs(),
    score_fn=score_fn,
    budget=BudgetConfig(dollars=10),
    sampler_model_pairs=[
        SamplerModelPair(sampler="softmax", model=LIGHT_MODEL, weight=1.0, temperature=0.3),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODEL, weight=1.0, temperature=0.7),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODEL, weight=1.0, temperature=1.0),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODEL, weight=1.0, temperature=1.2),
    ],
    cvt=CVTConfig(n_centroids=40, defer_centroids=True),
    init=InitConfig(
        enabled=True,
        n_diverse_seeds=3,
        n_variants_per_seed=5,
        diversity_model=HEAVY_MODEL,
        variant_models=[LIGHT_MODEL],
        temperature=0.8,
    ),
    meta_advice=MetaAdviceConfig(enabled=False, interval=50, model=HEAVY_MODEL),
    pipeline=PipelineConfig(n_llm_workers=8, n_eval_processes=8, n_inspirations=1, output_mode="diff"),
    behavior=BehaviorConfig(ast_features=PRISM_AST_FEATURES, score_keys=PRISM_SCORE_KEYS),
    punctuated_equilibrium=PunctuatedEquilibriumConfig(
        enabled=True,
        interval=10,
        n_clusters=3,
        n_variants=3,
        heavy_model=HEAVY_MODEL,
        variant_models=[LIGHT_MODEL],
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
