#!/usr/bin/env python3
"""Run AlgoForge on PRISM (GPU Model Placement) Problem."""

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
    budget_dollars=4.50,
    local_endpoints={"Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8001/v1"},
    behavior=af.BehaviorConfig(
        ast_features=['loop_nesting_max', 'branch_count', 'comparison_count', 'subscript_count'],
    ),
)
