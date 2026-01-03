#!/usr/bin/env python3
"""Test AlgoForge on Transaction Scheduling Problem."""

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
})

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3
from prompts import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM
from algoforge import run, AlgoforgeConfig, BudgetConfig, SamplerModelPair, InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig

# --- Problem Setup ---
WORKLOADS = [Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)]

# Scoring reference points: sequential ordering = 0 pts, theoretical optimal = 100 pts
BASELINE = sum(w.get_opt_seq_cost(list(range(w.num_txns))) for w in WORKLOADS)
OPTIMAL = sum(max(txn[0][3] for txn in w.txns) for w in WORKLOADS)
EFFECTIVE_OPTIMAL = OPTIMAL + 0.10 * (BASELINE - OPTIMAL)  # Shifted to make 100 achievable


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


# --- Config ---
LIGHT_MODELS = [
    "openrouter/qwen/qwen3-coder-30b-a3b-instruct",
    "openrouter/google/gemini-2.5-flash-lite",
    "openrouter/deepseek/deepseek-v3.2",
]
HEAVY_MODEL = "openrouter/deepseek/deepseek-v3.2"

RUN_DIR = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=WORKLOADS,
    score_fn=score_fn,
    budget=BudgetConfig(dollars=3.0),
    sampler_model_pairs=[
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=1.0),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=1.0),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[2], weight=1.0, temperature=1.0),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=2.0),
        SamplerModelPair(sampler="softmax", model=HEAVY_MODEL, weight=1.0, temperature=0.3),
    ],
    cvt=CVTConfig(n_centroids=50, defer_centroids=True),
    init=InitConfig(
        n_diverse_seeds=8,
        n_variants_per_seed=15,
        diversity_model="openrouter/z-ai/glm-4.7",
        variant_model=LIGHT_MODELS[1],  # gemini-2.5-flash-lite
    ),
    meta_advice=MetaAdviceConfig(interval=50, model=HEAVY_MODEL),
    pipeline=PipelineConfig(n_llm_workers=4, n_eval_processes=4, n_inspirations=1),
    output_dir=RUN_DIR,
)

# --- Run ---
if __name__ == "__main__":
    print(f"Budget: ${config.budget.dollars} | Baseline: {BASELINE} | Target: {EFFECTIVE_OPTIMAL:.0f}")
    print(f"Output: {RUN_DIR}/snapshot.json\n")
    result = run(config)
    print(f"\nBest: {result.best_score:.1f} pts | Evals: {result.total_evaluations} | Cost: ${result.total_cost:.4f}")
    print(f"Snapshot: {RUN_DIR}/snapshot.json")
    print(f"\n{result.best_program[:800]}{'...' if len(result.best_program) > 800 else ''}")
