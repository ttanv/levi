#!/usr/bin/env python3
"""Run AlgoForge for Transaction Scheduling."""

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, score_fn
import algoforge as af

result = af.evolve_code(
    PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    score_fn=score_fn,
    paradigm_model="openrouter/google/gemini-3-flash-preview",
    mutation_model=[
        "openrouter/xiaomi/mimo-v2-flash",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ],
    budget_dollars=13,
    local_endpoints={"Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8001/v1"},
    pipeline=af.PipelineConfig(eval_timeout=300),
)
