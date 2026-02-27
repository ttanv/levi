#!/usr/bin/env python3
"""Run AlgoForge for Can't Be Late Scheduling."""

from datetime import datetime

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, INPUTS, score_fn, DIVERSITY_SEED_PROMPT
import algoforge as af

result = af.evolve_code(
    PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    score_fn=score_fn,
    inputs=INPUTS,
    paradigm_model="openrouter/google/gemini-3-flash-preview",
    mutation_model=[
        "openrouter/xiaomi/mimo-v2-flash",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ],
    budget_dollars=5.00,
    local_endpoints={"Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8001/v1"},
    model_info={
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
    },
    init=af.InitConfig(
        n_diverse_seeds=5,
        n_variants_per_seed=20,
        diversity_prompt=DIVERSITY_SEED_PROMPT,
    ),
    pipeline=af.PipelineConfig(n_llm_workers=12, n_eval_processes=12, n_inspirations=1, eval_timeout=300),
    behavior=af.BehaviorConfig(
        ast_features=['cyclomatic_complexity', 'comparison_count', 'math_operators', 'branch_count'],
        score_keys=['tight_deadline_score', 'loose_deadline_score'],
    ),
    punctuated_equilibrium=af.PunctuatedEquilibriumConfig(
        enabled=True,
        interval=5,
        n_clusters=3,
        n_variants=3,
        temperature=0.7,
        reasoning_effort="low",
    ),
    prompt_opt=af.PromptOptConfig(enabled=True),
    output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_po",
)
