#!/usr/bin/env python3
"""Run AlgoForge with Prompt Optimization for LLM SQL (Column Reordering)."""

from datetime import datetime

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, INPUTS, score_fn
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig, 
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig,
    PunctuatedEquilibriumConfig, PromptOptConfig,
)

LLM_SQL_AST_FEATURES = ['loop_nesting_max', 'comparison_count', 'call_count', 'branch_count']

config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=INPUTS,
    score_fn=score_fn,
    paradigm_models="openrouter/google/gemini-3-flash-preview",
    mutation_models=[
        "openrouter/xiaomi/mimo-v2-flash",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ],
    local_endpoints={"Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8001/v1"},
    model_info={
        "xiaomi/mimo-v2-flash": {
            "input_cost_per_token": 0.00000009,
            "output_cost_per_token": 0.00000029,
        },
        "Qwen/Qwen3-30B-A3B-Instruct-2507": {
            "input_cost_per_token": 0.0000001,
            "output_cost_per_token": 0.0000004,
        },
    },
    budget=BudgetConfig(dollars=4.50),
    cvt=CVTConfig(n_centroids=100, defer_centroids=True),
    init=InitConfig(
        enabled=True,
        n_diverse_seeds=6,
        n_variants_per_seed=25,
        temperature=0.8,
    ),
    meta_advice=MetaAdviceConfig(enabled=True, interval=50),
    pipeline=PipelineConfig(n_llm_workers=12, n_eval_processes=12, n_inspirations=1, output_mode="full", eval_timeout=300),
    behavior=BehaviorConfig(ast_features=LLM_SQL_AST_FEATURES, score_keys=[], init_noise=0.0),
    punctuated_equilibrium=PunctuatedEquilibriumConfig(
        enabled=True,
        interval=10,
        n_clusters=3,
        n_variants=3,
        behavior_noise=0.0,
        temperature=0.7,
        reasoning_effort="low",
    ),
    prompt_opt=PromptOptConfig(enabled=True),
    output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_po",
)

if __name__ == "__main__":
    run(config)
