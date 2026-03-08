#!/usr/bin/env python3
"""Run Levi on n=26 circle packing with local Qwen + OpenRouter GPT-5."""

import json
import os
from datetime import datetime
from pathlib import Path

import levi
from problem import FUNCTION_SIGNATURE, PROBLEM_DESCRIPTION, SEED_PROGRAM, score_fn

QWEN_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PARADIGM_SHIFT_MODEL = "openrouter/google/gemini-3-flash-preview"
RESUME_SNAPSHOT_PATH = Path("runs/20260305_193804_circle_packing/snapshot.json")


def main() -> None:
    endpoint = os.getenv("LEVI_LOCAL_ENDPOINT", "http://localhost:8001/v1")
    budget = 15.0
    resume_snapshot = None
    output_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_circle_packing"

    if RESUME_SNAPSHOT_PATH.exists():
        with open(RESUME_SNAPSHOT_PATH) as f:
            resume_snapshot = json.load(f)
        output_dir = str(RESUME_SNAPSHOT_PATH.parent)
        print(f"Resuming from snapshot: {RESUME_SNAPSHOT_PATH}")
    else:
        print(f"Starting fresh run (snapshot not found): {RESUME_SNAPSHOT_PATH}")

    result = levi.evolve_code(
        PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        seed_program=SEED_PROGRAM,
        score_fn=score_fn,
        paradigm_model=PARADIGM_SHIFT_MODEL,
        mutation_model=[QWEN_MODEL, "openrouter/xiaomi/mimo-v2-flash"],
        local_endpoints={QWEN_MODEL: endpoint},
        model_info={
            QWEN_MODEL: {
                "input_cost_per_token": 0.0000001,
                "output_cost_per_token": 0.0000004,
            },
        },
        budget_dollars=budget,
        pipeline=levi.PipelineConfig(
            n_llm_workers=8,
            n_eval_processes=8,
            max_tokens=2048,
            eval_timeout=600,
        ),
        behavior=levi.BehaviorConfig(
            ast_features=[
                "function_def_count",
                "loop_nesting_max",
                "comparison_count",
                "range_max_arg",
            ],
            score_keys=[
                "boundary_touch_fraction",
                "nn_gap_cv",
                "radius_entropy",
                "execution_time",
            ],
        ),
        punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(
            reasoning_effort="low",
        ),
        init=levi.InitConfig(
            diversity_model=PARADIGM_SHIFT_MODEL,
        ),
        output_dir=output_dir,
        resume_snapshot=resume_snapshot,
    )

    print(f"Best score: {result.best_score:.17g}")
    print(result.best_program)


if __name__ == "__main__":
    main()
