#!/usr/bin/env python3
"""Run AlgoForge for EPLB."""

from datetime import datetime

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, get_lazy_inputs, score_fn, DIVERSITY_SEED_PROMPT
import algoforge as af

result = af.evolve_code(
    PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    score_fn=score_fn,
    inputs=get_lazy_inputs(),
    paradigm_model="openrouter/google/gemini-3-flash-preview",
    mutation_model=[
        "openrouter/xiaomi/mimo-v2-flash",
        "openrouter/qwen/qwen3-30b-a3b-instruct-2507",
    ],
    budget_dollars=4.50,
    model_info={"xiaomi/mimo-v2-flash": {
        "input_cost_per_token": 0.00000009,
        "output_cost_per_token": 0.00000029,
    }},
    init=af.InitConfig(
        n_diverse_seeds=5,
        n_variants_per_seed=20,
        diversity_prompt=DIVERSITY_SEED_PROMPT,
    ),
    pipeline=af.PipelineConfig(n_llm_workers=12, n_eval_processes=12, n_inspirations=1, eval_timeout=300),
    behavior=af.BehaviorConfig(
        ast_features=['loop_nesting_max', 'cyclomatic_complexity', 'math_operators'],
        score_keys=['execution_time', 'workload_main', 'workload_8', 'workload_9'],
        init_noise=0.3,
    ),
    punctuated_equilibrium=af.PunctuatedEquilibriumConfig(
        enabled=True,
        interval=5,
        n_clusters=3,
        n_variants=3,
        behavior_noise=0.3,
        temperature=0.7,
        reasoning_effort="low",
    ),
    prompt_opt=af.PromptOptConfig(enabled=True),
    output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_po",
)
