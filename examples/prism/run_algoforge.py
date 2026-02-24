#!/usr/bin/env python3
"""Run AlgoForge on PRISM (GPU Model Placement) Problem."""

from datetime import datetime

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, get_lazy_inputs, score_fn
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig,
    PipelineConfig, CVTConfig, BehaviorConfig, PunctuatedEquilibriumConfig,
)

# Behavioral dimensions: AST features that distinguish algorithmic approaches
PRISM_AST_FEATURES = ['loop_nesting_max', 'branch_count', 'comparison_count', 'subscript_count']

config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=get_lazy_inputs(),
    score_fn=score_fn,
    paradigm_models="openrouter/google/gemini-3-flash-preview",
    mutation_models=[
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ],
    local_endpoints={"Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8001/v1"},
    model_info={"Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000004,
    }},
    budget=BudgetConfig(dollars=5),
    cvt=CVTConfig(n_centroids=50, defer_centroids=True),
    pipeline=PipelineConfig(n_llm_workers=8, n_eval_processes=8, n_inspirations=1, output_mode="diff"),
    behavior=BehaviorConfig(ast_features=PRISM_AST_FEATURES, score_keys=[], init_noise=0.0),
    punctuated_equilibrium=PunctuatedEquilibriumConfig(
        enabled=True,
        interval=10,
        n_clusters=3,
        n_variants=3,
        behavior_noise=0.0,
        temperature=0.7,
        reasoning_effort="disabled",
    ),
    output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)

if __name__ == "__main__":
    run(config)
