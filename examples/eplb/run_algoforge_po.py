#!/usr/bin/env python3
"""
Run AlgoForge with Prompt Optimization for EPLB.

Workflow:
1. Check for cached optimized prompts (optimized_prompts.json)
2. If not cached, run DSPy optimization (~$0.60)
3. Run AlgoForge with optimized prompts
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add algoforge to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, get_lazy_inputs, score_fn, DIVERSITY_SEED_PROMPT
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig, SamplerModelPair,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig
)
from algoforge.config.models import PunctuatedEquilibriumConfig

# --- Constants ---
OPTIMIZED_PROMPTS_FILE = Path(__file__).parent / "optimized_prompts.json"
OPTIMIZATION_BUDGET = 0.60
MAIN_BUDGET = 4.50

# Behavioral dimensions
EPLB_AST_FEATURES = ['loop_nesting_max', 'cyclomatic_complexity', 'math_operators']
EPLB_SCORE_KEYS = ['execution_time', 'workload_main', 'workload_8', 'workload_9']

# Models
LIGHT_MODELS = [
    "openrouter/xiaomi/mimo-v2-flash",
    "openrouter/qwen/qwen3-30b-a3b-instruct-2507",
]
PARADIGM_SHIFT_MODEL = "openrouter/google/gemini-3-flash-preview"

LOCAL_ENDPOINTS = {}

MODEL_INFO = {
    "xiaomi/mimo-v2-flash": {
        "max_tokens": 16384,
        "max_input_tokens": 262144,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000009,
        "output_cost_per_token": 0.00000029,
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
    "openrouter/qwen/qwen3-30b-a3b-instruct-2507": "qwen",
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
        inputs=get_lazy_inputs(),
        score_fn=score_fn,
        budget=BudgetConfig(dollars=MAIN_BUDGET),
        sampler_model_pairs=[
            # MiMo-V2-Flash (OpenRouter) - cyclic annealing, 1 big cycle
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[0], weight=1.0, n_cycles=1),
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[0], weight=1.0, n_cycles=1),
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[0], weight=1.0, n_cycles=1),
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[0], weight=1.0, n_cycles=1),
            # Qwen 30B (OpenRouter) - cyclic annealing, 1 big cycle
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[1], weight=1.0, n_cycles=1),
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[1], weight=1.0, n_cycles=1),
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[1], weight=1.0, n_cycles=1),
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[1], weight=1.0, n_cycles=1),
            # Gemini Flash (OpenRouter) - paradigm shift model
            SamplerModelPair(sampler="cyclic_annealing", model=PARADIGM_SHIFT_MODEL, weight=1.0, n_cycles=1),
        ],
        cvt=CVTConfig(
            n_centroids=50,
            defer_centroids=True,
        ),
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
        behavior=BehaviorConfig(ast_features=EPLB_AST_FEATURES, score_keys=EPLB_SCORE_KEYS, init_noise=0.3),
        punctuated_equilibrium=PunctuatedEquilibriumConfig(
            enabled=True,
            interval=5,
            n_clusters=3,
            n_variants=3,
            heavy_models=[PARADIGM_SHIFT_MODEL],
            variant_models=LIGHT_MODELS,
            behavior_noise=0.3,
            temperature=0.7,
            reasoning_effort="low",
        ),
        output_dir=run_dir,
        local_endpoints=LOCAL_ENDPOINTS,
        model_info=MODEL_INFO,
        prompt_overrides=prompt_overrides,
    )

    return config


def find_latest_run() -> Path | None:
    """Find the latest run directory under runs/ with a snapshot.json."""
    runs_dir = Path(__file__).parent / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and (d / "snapshot.json").exists()],
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlgoForge with Prompt Optimization - EPLB")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest run in runs/")
    args = parser.parse_args()

    print("="*60)
    print("AlgoForge with Prompt Optimization - EPLB")
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

    # Check for resume
    resume_snapshot = None
    if args.resume:
        latest_run = find_latest_run()
        if latest_run is None:
            print("No previous run found to resume from. Starting fresh.")
        else:
            snapshot_path = latest_run / "snapshot.json"
            with open(snapshot_path) as f:
                resume_snapshot = json.load(f)
            run_state = resume_snapshot.get("run_state", {})
            print(f"Resuming from: {latest_run.name}")
            print(f"  Prior cost: ${run_state.get('total_cost', 0):.3f}")
            print(f"  Prior evals: {run_state.get('eval_count', 0)}")
            print(f"  Prior best score: {run_state.get('best_score', 0):.1f}")
            print()

    # Build config and run
    config = build_config_with_optimized_prompts(optimized)

    # When resuming, reuse the same output directory
    if resume_snapshot and latest_run:
        config.output_dir = str(latest_run)

    print(f"Output directory: {config.output_dir}")
    print()

    run(config, resume_snapshot=resume_snapshot)
