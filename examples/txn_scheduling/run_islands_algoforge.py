#!/usr/bin/env python3
"""
Island-based AlgoForge on Transaction Scheduling Problem.

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
    "openrouter/z-ai/glm-4.7": {
        "max_tokens": 32768,
        "max_input_tokens": 202752,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0000004,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "openrouter",
    },
    "openrouter/google/gemini-2.5-pro-preview-05-06": {
        "max_tokens": 65536,
        "max_input_tokens": 1048576,
        "max_output_tokens": 65536,
        "input_cost_per_token": 0.00000125,
        "output_cost_per_token": 0.00001,
        "litellm_provider": "openrouter",
    },
})

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3
from prompts import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM
from algoforge import AlgoforgeConfig, BudgetConfig, SamplerModelPair, InitConfig, PipelineConfig, CVTConfig, MetaAdviceConfig
from algoforge.island import run_islands

# --- Problem Setup ---
WORKLOADS = [Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)]

# Scoring reference points
BASELINE = sum(w.get_opt_seq_cost(list(range(w.num_txns))) for w in WORKLOADS)
OPTIMAL = sum(max(txn[0][3] for txn in w.txns) for w in WORKLOADS)
EFFECTIVE_OPTIMAL = OPTIMAL + 0.10 * (BASELINE - OPTIMAL)


def score_fn(get_best_schedule, inputs):
    """Evaluate scheduling algorithm: returns 0-100 score based on total makespan."""
    try:
        total = 0
        for w in inputs:
            _, schedule = get_best_schedule(w, 10)
            if set(schedule) != set(range(w.num_txns)):
                return {"error": "Invalid schedule: not a permutation"}
            total += w.get_opt_seq_cost(schedule)

        if total >= BASELINE:
            score = 0.0
        elif total <= EFFECTIVE_OPTIMAL:
            score = 100.0
        else:
            score = ((BASELINE - total) / (BASELINE - EFFECTIVE_OPTIMAL)) * 100

        return {"score": score, "makespan": total}
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
N_ISLANDS = 3
CULLING_CHECKPOINTS = [0.25, 0.50, 0.75]  # Cull bottom half of islands at these budget %
BUDGET_USD = 5.0

RUN_DIR = f"runs/islands_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# --- AlgoForge Config ---
config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=WORKLOADS,
    score_fn=score_fn,
    budget=BudgetConfig(dollars=BUDGET_USD),
    sampler_model_pairs=[
        SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[0], weight=1.0),
        SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[1], weight=1.0),
        SamplerModelPair(sampler="cyclic_annealing", model=LIGHT_MODELS[2], weight=1.0),
        SamplerModelPair(sampler="cyclic_annealing", model=HEAVY_MODEL, weight=1.0),
    ],
    cvt=CVTConfig(n_centroids=20, defer_centroids=True),
    init=InitConfig(
        n_diverse_seeds=N_ISLANDS,  # One diverse seed per island
        n_variants_per_seed=30,
        diversity_model="openrouter/google/gemini-2.5-pro-preview-05-06",
        variant_model=LIGHT_MODELS[1],
    ),
    meta_advice=MetaAdviceConfig(enabled=True, interval=50, model=HEAVY_MODEL),
    pipeline=PipelineConfig(n_llm_workers=4, n_eval_processes=8, n_inspirations=1),
    output_dir=RUN_DIR,
)

# --- Run ---
if __name__ == "__main__":
    print(f"Islands: {N_ISLANDS} | Budget: ${BUDGET_USD} | Culling at: {CULLING_CHECKPOINTS}")
    print(f"Baseline: {BASELINE} | Target: {EFFECTIVE_OPTIMAL:.0f}")
    print(f"Output: {RUN_DIR}/snapshot.json\n")

    result = run_islands(
        config,
        n_islands=N_ISLANDS,
        culling_checkpoints=CULLING_CHECKPOINTS,
    )

    print(f"\nBest: {result.best_score:.1f} pts | Evals: {result.total_evaluations} | Cost: ${result.total_cost:.4f}")
    print(f"Total archive size: {result.archive_size}")
    print(f"Snapshot: {RUN_DIR}/snapshot.json")
    print(f"\n{result.best_program[:800]}{'...' if len(result.best_program) > 800 else ''}")
