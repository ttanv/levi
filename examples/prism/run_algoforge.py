#!/usr/bin/env python3
"""Run AlgoForge on PRISM (GPU Model Placement) Problem."""

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
    mutation_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    budget_dollars=5.0,
    local_endpoints={"Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8001/v1"},
    model_info={"Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000004,
    }},
    pipeline=af.PipelineConfig(n_llm_workers=8, n_eval_processes=8, n_inspirations=1, output_mode="diff"),
    behavior=af.BehaviorConfig(
        ast_features=['loop_nesting_max', 'branch_count', 'comparison_count', 'subscript_count'],
    ),
    punctuated_equilibrium=af.PunctuatedEquilibriumConfig(
        enabled=True,
        interval=10,
        n_clusters=3,
        n_variants=3,
        temperature=0.7,
        reasoning_effort="disabled",
    ),
    output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
