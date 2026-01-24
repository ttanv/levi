#!/usr/bin/env python3
"""Run AlgoForge on PRISM (GPU Model Placement) Problem."""

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
# Light models: OpenRouter MiMo-V2-Flash + DeepSeek + Local Qwen 30B
LIGHT_MODELS = [
    "openrouter/xiaomi/mimo-v2-flash",
    "openrouter/deepseek/deepseek-v3.2",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
]
PARADIGM_SHIFT_MODEL = "openrouter/google/gemini-3-flash-preview"

# Model -> API base URL mapping for local TPU endpoints
LOCAL_ENDPOINTS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8001/v1",
}

# Model info for token cost tracking (same format as litellm.register_model)
MODEL_INFO = {
    "xiaomi/mimo-v2-flash": {
        "max_tokens": 16384,
        "max_input_tokens": 262144,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000009,   # $0.09/M input
        "output_cost_per_token": 0.00000029,  # $0.29/M output
    },
    "deepseek/deepseek-v3.2": {
        "max_tokens": 16384,
        "max_input_tokens": 163840,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000025,   # $0.25/M input
        "output_cost_per_token": 0.00000038,  # $0.38/M output
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000004,
    },
    "google/gemini-3-flash-preview": {
        "max_tokens": 65536,
        "max_input_tokens": 1048576,
        "max_output_tokens": 65536,
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
    budget=BudgetConfig(dollars=5),
    sampler_model_pairs=[
        # MiMo-V2-Flash (OpenRouter)
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=0.3),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=0.7),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=1.0),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=1.2),
        # DeepSeek (OpenRouter)
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=0.3),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=0.7),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=1.0),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=1.2),
        # Qwen 30B (Local TPU)
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[2], weight=1.0, temperature=0.3),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[2], weight=1.0, temperature=0.7),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[2], weight=1.0, temperature=1.0),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[2], weight=1.0, temperature=1.2),
    ],
    cvt=CVTConfig(n_centroids=40, defer_centroids=True, predefined_centroids_file="/home/ttanveer/algoforge/examples/prism/centroids.json"),
    init=InitConfig(
        enabled=True,
        n_diverse_seeds=3,
        n_variants_per_seed=5,
        diversity_model=PARADIGM_SHIFT_MODEL,
        variant_models=LIGHT_MODELS,
        temperature=0.8,
    ),
    meta_advice=MetaAdviceConfig(enabled=False, interval=50, model=PARADIGM_SHIFT_MODEL),
    pipeline=PipelineConfig(n_llm_workers=8, n_eval_processes=8, n_inspirations=1, output_mode="diff"),
    behavior=BehaviorConfig(ast_features=PRISM_AST_FEATURES, score_keys=PRISM_SCORE_KEYS, init_noise=0.0),
    punctuated_equilibrium=PunctuatedEquilibriumConfig(
        enabled=True,
        interval=10,
        n_clusters=3,
        n_variants=3,
        heavy_model=PARADIGM_SHIFT_MODEL,
        variant_models=LIGHT_MODELS,
        behavior_noise=0.0,
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
