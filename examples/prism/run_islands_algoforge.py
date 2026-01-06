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

from prism_problem import (
    TEST_CASES, GPU_MEM_SIZE,
    calculate_kvpr, round_robin_placement, compute_theoretical_optimal
)
from prompts import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM
from algoforge import AlgoforgeConfig, BudgetConfig, SamplerModelPair, InitConfig, PipelineConfig, CVTConfig, MetaAdviceConfig
from algoforge.island import run_islands


def score_fn(compute_model_placement, inputs):
    """Evaluate placement algorithm: returns 0-100 score with sqrt scaling."""
    try:
        all_scores = []

        for gpu_num, models in inputs:
            # Run solution
            result = compute_model_placement(gpu_num, models)

            # Validate return type
            if not isinstance(result, dict):
                return {"error": f"Expected dict, got {type(result).__name__}"}

            # Validate all models placed exactly once
            placed = []
            for gpu_id, gpu_models in result.items():
                if not isinstance(gpu_models, list):
                    return {"error": f"GPU {gpu_id} value must be list, got {type(gpu_models).__name__}"}
                placed.extend(gpu_models)

            if len(placed) != len(models):
                return {"error": f"Not all models placed: {len(placed)}/{len(models)}"}

            # Validate memory constraints
            for gpu_id, gpu_models in result.items():
                total_size = sum(m.model_size for m in gpu_models)
                if total_size > GPU_MEM_SIZE:
                    return {"error": f"GPU {gpu_id} exceeds memory: {total_size}GB > {GPU_MEM_SIZE}GB"}

            # Compute scores
            baseline_kvpr = calculate_kvpr(round_robin_placement(gpu_num, models))
            optimal_kvpr = compute_theoretical_optimal(gpu_num, models)
            solution_kvpr = calculate_kvpr(result)

            # Score with sqrt scaling
            if baseline_kvpr > optimal_kvpr:
                raw_ratio = (baseline_kvpr - solution_kvpr) / (baseline_kvpr - optimal_kvpr)
                test_score = 100.0 * (max(0.0, min(1.0, raw_ratio)) ** 0.5)
            else:
                test_score = 100.0 if solution_kvpr <= optimal_kvpr else 0.0

            all_scores.append(test_score)

        return {"score": sum(all_scores) / len(all_scores), "num_tests": len(all_scores)}

    except Exception as e:
        return {"error": str(e)}


# --- Model Config ---
LIGHT_MODELS = [
    "openrouter/qwen/qwen3-coder-30b-a3b-instruct",
    "openrouter/google/gemini-2.5-flash-lite",
    "openrouter/deepseek/deepseek-v3.2",
]
HEAVY_MODEL = "openrouter/deepseek/deepseek-v3.2"

# --- Island Config ---
N_ISLANDS = 4
CULLING_CHECKPOINTS = [0.5]  # Cull at 50% budget
MIGRATION_INTERVAL = 999999999  # Effectively disable migration
BUDGET_USD = 3.0

RUN_DIR = f"runs/islands_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# --- AlgoForge Config ---
config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=TEST_CASES,
    score_fn=score_fn,
    budget=BudgetConfig(dollars=BUDGET_USD),
    sampler_model_pairs=[
        SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[0], weight=1.0),
        SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[1], weight=1.0),
        SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[2], weight=1.0),
        SamplerModelPair(sampler="cyclic_annealing", model=HEAVY_MODEL, weight=1.0),
    ],
    cvt=CVTConfig(n_centroids=15, defer_centroids=True),
    init=InitConfig(
        n_diverse_seeds=8,  # Generate 8 seeds, best 4 become islands
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
    # Test seed program first
    exec_globals = {}
    exec(SEED_PROGRAM, exec_globals)
    seed_result = score_fn(exec_globals['compute_model_placement'], TEST_CASES)
    print(f"Seed program score: {seed_result}")

    print(f"\nIslands: {N_ISLANDS} (from 8 seeds) | Budget: ${BUDGET_USD} | Culling at: {CULLING_CHECKPOINTS}")
    print(f"Centroids: 15 | Test cases: {len(TEST_CASES)}")
    print(f"Output: {RUN_DIR}/snapshot.json\n")

    result = run_islands(
        config,
        n_islands=N_ISLANDS,
        culling_checkpoints=CULLING_CHECKPOINTS,
        migration_interval=MIGRATION_INTERVAL,
    )

    print(f"\nBest: {result.best_score:.1f} pts | Evals: {result.total_evaluations} | Cost: ${result.total_cost:.4f}")
    print(f"Total archive size: {result.archive_size}")
    print(f"Snapshot: {RUN_DIR}/snapshot.json")
    print(f"\n{result.best_program[:800]}{'...' if len(result.best_program) > 800 else ''}")
