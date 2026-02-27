#!/usr/bin/env python3
"""Run AlgoForge with Prompt Optimization for Transaction Scheduling."""

from datetime import datetime

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, INPUTS, score_fn, DIVERSITY_SEED_PROMPT
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig,
    PunctuatedEquilibriumConfig, PromptOptConfig,
)

TXN_AST_FEATURES = ['loop_nesting_max', 'comparison_count', 'math_operators', 'branch_count']

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
    budget=BudgetConfig(dollars=19.10),
    cvt=CVTConfig(n_centroids=50, defer_centroids=True),
    init=InitConfig(
        enabled=True,
        n_diverse_seeds=4,
        n_variants_per_seed=20,
        temperature=0.8,
        diversity_prompt=DIVERSITY_SEED_PROMPT,
    ),
    meta_advice=MetaAdviceConfig(enabled=True, interval=50),
    pipeline=PipelineConfig(n_llm_workers=1, n_eval_processes=1, n_inspirations=1, output_mode="full", eval_timeout=300),
    behavior=BehaviorConfig(ast_features=TXN_AST_FEATURES, score_keys=[], init_noise=0.2),
    punctuated_equilibrium=PunctuatedEquilibriumConfig(
        enabled=True,
        interval=10,
        n_clusters=3,
        n_variants=3,
        behavior_noise=0.0,
        temperature=0.7,
        reasoning_effort="disabled",
    ),
    prompt_opt=PromptOptConfig(enabled=False),
    output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_po",
)

if __name__ == "__main__":
    run(config)
