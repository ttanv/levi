#!/usr/bin/env python3
"""Run AlgoForge with Prompt Optimization for Transaction Scheduling."""

import json
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

# Behavioral dimensions
TXN_AST_FEATURES = ['loop_nesting_max', 'comparison_count', 'call_count', 'branch_count']

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


def find_latest_run() -> Path | None:
    """Find the latest run directory under runs/ with a snapshot.json."""
    runs_dir = Path(__file__).parent.parent.parent / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and (d / "snapshot.json").exists() and d.name.endswith("_po")],
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def build_config() -> AlgoforgeConfig:
    run_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_po"

    return AlgoforgeConfig(
        problem_description=PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        seed_program=SEED_PROGRAM,
        inputs=INPUTS,
        score_fn=score_fn,
        budget=BudgetConfig(dollars=19.10),
        sampler_model_pairs=[
            # MiMo-V2-Flash (OpenRouter) - cyclic annealing
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[0], weight=1.0, n_cycles=1),
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[0], weight=1.0, n_cycles=1),
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[0], weight=1.0, n_cycles=1),
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[0], weight=1.0, n_cycles=1),
            # Qwen 30B (Local TPU) - cyclic annealing
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[1], weight=1.0, n_cycles=1),
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[1], weight=1.0, n_cycles=1),
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[1], weight=1.0, n_cycles=1),
            SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[1], weight=1.0, n_cycles=1),
            # Gemini Flash (OpenRouter) - paradigm shift model
            SamplerModelPair(sampler="cyclic_annealing", model=PARADIGM_SHIFT_MODEL, weight=1.0, n_cycles=1),
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
        behavior=BehaviorConfig(ast_features=TXN_AST_FEATURES, score_keys=[], init_noise=0.3),
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
        prompt_opt=PromptOptConfig(enabled=True),
        output_dir=run_dir,
        local_endpoints=LOCAL_ENDPOINTS,
        model_info=MODEL_INFO,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlgoForge with Prompt Optimization - TXN Scheduling")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest run in runs/")
    args = parser.parse_args()

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

    config = build_config()

    if resume_snapshot and latest_run:
        config.output_dir = str(latest_run)

    run(config, resume_snapshot=resume_snapshot)
