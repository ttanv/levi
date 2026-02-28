#!/usr/bin/env python3
"""Run AlgoForge for EPLB."""

from datetime import datetime

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, get_lazy_inputs, score_fn
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
    pipeline=af.PipelineConfig(n_llm_workers=12, n_eval_processes=12, n_inspirations=1, eval_timeout=300),
    behavior=af.BehaviorConfig(
        ast_features=['loop_nesting_max', 'cyclomatic_complexity', 'math_operators'],
        score_keys=['execution_time', 'workload_main', 'workload_8', 'workload_9'],
    ),
    prompt_opt=af.PromptOptConfig(enabled=True),
    output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_po",
)
