#!/usr/bin/env python3
"""Run AlgoForge with Prompt Optimization for EPLB."""

from datetime import datetime

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, get_lazy_inputs, score_fn, DIVERSITY_SEED_PROMPT
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig, SamplerModelPair,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig,
    PunctuatedEquilibriumConfig, PromptOptConfig,
)

EPLB_AST_FEATURES = ['loop_nesting_max', 'cyclomatic_complexity', 'math_operators']
EPLB_SCORE_KEYS = ['execution_time', 'workload_main', 'workload_8', 'workload_9']

config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=get_lazy_inputs(),
    score_fn=score_fn,
    paradigm_models="openrouter/google/gemini-3-flash-preview",
    mutation_models=[
        "openrouter/xiaomi/mimo-v2-flash",
        "openrouter/qwen/qwen3-30b-a3b-instruct-2507",
    ],
    model_info={"xiaomi/mimo-v2-flash": {
        "input_cost_per_token": 0.00000009,
        "output_cost_per_token": 0.00000029,
    }},
    budget=BudgetConfig(dollars=4.50),
    cvt=CVTConfig(n_centroids=50, defer_centroids=True),
    init=InitConfig(
        enabled=True,
        n_diverse_seeds=5,
        n_variants_per_seed=20,
        temperature=0.8,
        diversity_prompt=DIVERSITY_SEED_PROMPT,
    ),
    meta_advice=MetaAdviceConfig(enabled=True, interval=50),
    pipeline=PipelineConfig(n_llm_workers=12, n_eval_processes=12, n_inspirations=1, output_mode="full", eval_timeout=300),
    behavior=BehaviorConfig(ast_features=EPLB_AST_FEATURES, score_keys=EPLB_SCORE_KEYS, init_noise=0.3),
    punctuated_equilibrium=PunctuatedEquilibriumConfig(
        enabled=True,
        interval=5,
        n_clusters=3,
        n_variants=3,
        behavior_noise=0.3,
        temperature=0.7,
        reasoning_effort="low",
    ),
    prompt_opt=PromptOptConfig(enabled=True),
    output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_po",
)

if __name__ == "__main__":
    run(config)
