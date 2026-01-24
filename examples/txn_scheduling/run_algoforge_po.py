#!/usr/bin/env python3
"""
Run AlgoForge with Prompt Optimization for Transaction Scheduling.

Workflow:
1. Check for cached optimized prompts (optimized_prompts.json)
2. If not cached, run DSPy optimization (~$0.60)
3. Run AlgoForge with optimized prompts (~$4.40)

Total budget: $5.00
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add algoforge to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, INPUTS, score_fn
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig, SamplerModelPair,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig
)
from algoforge.config.models import PunctuatedEquilibriumConfig, LLMProviderConfig

# --- Constants ---
OPTIMIZED_PROMPTS_FILE = Path(__file__).parent / "optimized_prompts.json"
OPTIMIZATION_BUDGET = 0.60
MAIN_BUDGET = 4.40

# Behavioral dimensions
TXN_AST_FEATURES = ['loop_nesting_max', 'comparison_count', 'call_count', 'branch_count']
TXN_SCORE_KEYS = []

# Models
LIGHT_MODELS = [
    "openrouter/xiaomi/mimo-v2-flash",
    "openrouter/deepseek/deepseek-v3.2",
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
    "deepseek/deepseek-v3.2": {
        "max_tokens": 16384,
        "max_input_tokens": 163840,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000038,
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

# Model key mapping for prompt overrides
MODEL_KEY_MAP = {
    "openrouter/xiaomi/mimo-v2-flash": "mimo",
    "openrouter/deepseek/deepseek-v3.2": "deepseek",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "qwen",
}


def load_or_run_optimization() -> dict:
    """Load cached prompts or run optimization."""
    if OPTIMIZED_PROMPTS_FILE.exists():
        print(f"Loading cached optimized prompts from: {OPTIMIZED_PROMPTS_FILE}")
        with open(OPTIMIZED_PROMPTS_FILE) as f:
            return json.load(f)

    print("No cached prompts found. Running optimization...")
    from optimize_prompts import run_optimization, save_optimized_prompts

    results = run_optimization(budget=OPTIMIZATION_BUDGET)
    save_optimized_prompts(results)

    with open(OPTIMIZED_PROMPTS_FILE) as f:
        return json.load(f)


def build_prompt_overrides(optimized: dict) -> dict:
    """
    Build prompt_overrides dict for AlgoforgeConfig.

    Structure:
    {
        "mutation": {
            "openrouter/xiaomi/mimo-v2-flash": "optimized instruction...",
            "openrouter/deepseek/deepseek-v3.2": "optimized instruction...",
            "Qwen/Qwen3-30B-A3B-Instruct-2507": "optimized instruction...",
        },
        "paradigm_shift": "optimized instruction..."
    }
    """
    overrides = {
        "mutation": {},
        "paradigm_shift": None,
    }

    # Map model keys back to full model names
    mutation_prompts = optimized.get("mutation", {})
    for model, key in MODEL_KEY_MAP.items():
        if key in mutation_prompts:
            overrides["mutation"][model] = mutation_prompts[key]

    if optimized.get("paradigm_shift"):
        overrides["paradigm_shift"] = optimized["paradigm_shift"]

    return overrides


def build_config_with_optimized_prompts(optimized: dict) -> AlgoforgeConfig:
    """Build AlgoforgeConfig with prompt_overrides field."""
    prompt_overrides = build_prompt_overrides(optimized)

    run_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_po"

    config = AlgoforgeConfig(
        problem_description=PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        seed_program=SEED_PROGRAM,
        inputs=INPUTS,
        score_fn=score_fn,
        budget=BudgetConfig(dollars=MAIN_BUDGET),
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
        cvt=CVTConfig(
            n_centroids=40,
            defer_centroids=True,
            predefined_centroids_file="/home/ttanveer/algoforge/examples/txn_scheduling/centroids.json"
        ),
        init=InitConfig(
            enabled=False,
            n_diverse_seeds=3,
            n_variants_per_seed=5,
            diversity_model=PARADIGM_SHIFT_MODEL,
            variant_models=LIGHT_MODELS,
            temperature=0.8,
        ),
        meta_advice=MetaAdviceConfig(enabled=False, interval=50, model=PARADIGM_SHIFT_MODEL),
        pipeline=PipelineConfig(n_llm_workers=8, n_eval_processes=8, n_inspirations=1, output_mode="full", eval_timeout=300),
        behavior=BehaviorConfig(ast_features=TXN_AST_FEATURES, score_keys=TXN_SCORE_KEYS, init_noise=0.0),
        punctuated_equilibrium=PunctuatedEquilibriumConfig(
            enabled=True,
            interval=5,
            n_clusters=3,
            n_variants=0,
            heavy_model=PARADIGM_SHIFT_MODEL,
            variant_models=LIGHT_MODELS,
            behavior_noise=0.0,
            temperature=0.7,
            reasoning_effort="disabled",
        ),
        output_dir=run_dir,
        llm=LLMProviderConfig(
            local_endpoints=LOCAL_ENDPOINTS,
            model_info=MODEL_INFO,
        ),
        prompt_overrides=prompt_overrides,
    )

    return config


if __name__ == "__main__":
    print("="*60)
    print("AlgoForge with Prompt Optimization - TXN Scheduling")
    print("="*60)
    print(f"Optimization budget: ${OPTIMIZATION_BUDGET:.2f}")
    print(f"Main run budget: ${MAIN_BUDGET:.2f}")
    print()

    # Load or run optimization
    optimized = load_or_run_optimization()

    # Show what we loaded
    mutation_count = len(optimized.get("mutation", {}))
    has_paradigm = optimized.get("paradigm_shift") is not None
    print(f"Loaded prompts: {mutation_count} mutation, paradigm_shift={has_paradigm}")
    print()

    # Build config and run
    config = build_config_with_optimized_prompts(optimized)
    print(f"Output directory: {config.output_dir}")
    print()

    run(config)
