#!/usr/bin/env python3
"""
Run Multi-Island Punctuated Equilibrium for Transaction Scheduling.

This implements a quick-and-dirty multi-island approach where:
- 4 islands share the same centroids.json behavior map
- Each island seeded with 1 unique LLM-generated seed (no variants)
- Evolution happens independently on all islands
- Every 5 evals: cross-island PE compares best from each island,
  accepts only if it beats the weakest island's best
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add algoforge to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, INPUTS, score_fn
from algoforge import AlgoforgeConfig, BudgetConfig, SamplerModelPair
from algoforge.config import InitConfig, PipelineConfig, CVTConfig, BehaviorConfig
from algoforge.config.models import PunctuatedEquilibriumConfig, LLMProviderConfig
from algoforge.island import run_multi_island_pe

# --- Constants ---
CENTROIDS_FILE = Path(__file__).parent / "centroids.json"
OPTIMIZED_PROMPTS_FILE = Path(__file__).parent / "optimized_prompts.json"
BUDGET = 5.00

# Behavioral dimensions (must match centroids.json)
TXN_AST_FEATURES = ['loop_nesting_max', 'comparison_count', 'call_count', 'branch_count']

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


def load_optimized_prompts() -> dict:
    """Load optimized prompts from JSON file."""
    if OPTIMIZED_PROMPTS_FILE.exists():
        print(f"Loading optimized prompts from: {OPTIMIZED_PROMPTS_FILE}")
        with open(OPTIMIZED_PROMPTS_FILE) as f:
            return json.load(f)
    print("Warning: No optimized_prompts.json found, using defaults")
    return {}


def build_prompt_overrides(optimized: dict) -> dict:
    """Build prompt_overrides dict for AlgoforgeConfig."""
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


def build_config() -> AlgoforgeConfig:
    """Build AlgoforgeConfig for multi-island PE."""
    run_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_mipe"

    # Load optimized prompts
    optimized = load_optimized_prompts()
    prompt_overrides = build_prompt_overrides(optimized)

    config = AlgoforgeConfig(
        problem_description=PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        seed_program=SEED_PROGRAM,
        inputs=INPUTS,
        score_fn=score_fn,
        budget=BudgetConfig(dollars=BUDGET),
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
        ),
        init=InitConfig(
            enabled=True,
            n_diverse_seeds=3,  # One per island
            n_variants_per_seed=0,  # No variants
            diversity_model=PARADIGM_SHIFT_MODEL,
            variant_models=LIGHT_MODELS,
            temperature=0.8,
        ),
        pipeline=PipelineConfig(
            n_llm_workers=8,
            n_eval_processes=8,
            n_inspirations=1,
            output_mode="full",
            eval_timeout=300,
        ),
        behavior=BehaviorConfig(
            ast_features=TXN_AST_FEATURES,
            score_keys=[],
            init_noise=0.0,
        ),
        punctuated_equilibrium=PunctuatedEquilibriumConfig(
            enabled=True,
            interval=5,  # Will be used as pe_interval
            n_clusters=3,
            n_variants=0,  # No variants for cross-island PE
            heavy_model=PARADIGM_SHIFT_MODEL,
            variant_models=LIGHT_MODELS,
            behavior_noise=0.0,
            temperature=0.7,
            reasoning_effort="low",
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
    print("Multi-Island Punctuated Equilibrium - TXN Scheduling")
    print("="*60)
    print(f"Budget: ${BUDGET:.2f}")
    print(f"Islands: 3")
    print(f"PE interval: 5 evals")
    print(f"Centroids: {CENTROIDS_FILE}")
    print()

    config = build_config()
    print(f"Output directory: {config.output_dir}")
    print()

    result = run_multi_island_pe(
        config=config,
        centroids_file=str(CENTROIDS_FILE),
        n_islands=3,
        pe_interval=5,
    )

    print()
    print("="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Best score: {result.best_score:.1f}")
    print(f"Total evaluations: {result.total_evaluations}")
    print(f"Total cost: ${result.total_cost:.3f}")
    print(f"Archive size: {result.archive_size}")
    print()
    print("Best program:")
    print(result.best_program[:1000])
    if len(result.best_program) > 1000:
        print("...")
