#!/usr/bin/env python3
"""Run Levi on n=26 circle packing with local Qwen 30B."""

import os
from datetime import datetime

import levi
from problem import FUNCTION_SIGNATURE, PROBLEM_DESCRIPTION, SEED_PROGRAM, score_fn

QWEN_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
GEMINI_FLASH_3_MODEL = "openrouter/google/gemini-3-flash-preview"
GEMINI_FLASH_LITE_PREVIEW_MODEL = "openrouter/google/gemini-3.1-flash-lite-preview"


def main() -> None:
    endpoint = os.getenv("LEVI_LOCAL_ENDPOINT", "http://localhost:8001/v1")
    budget = 6.0

    result = levi.evolve_code(
        PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        seed_program=SEED_PROGRAM,
        score_fn=score_fn,
        paradigm_model=[QWEN_MODEL, GEMINI_FLASH_3_MODEL],
        mutation_model=[QWEN_MODEL, GEMINI_FLASH_LITE_PREVIEW_MODEL],
        local_endpoints={QWEN_MODEL: endpoint},
        model_info={
            QWEN_MODEL: {
                "input_cost_per_token": 0.0000001,
                "output_cost_per_token": 0.0000004,
            },
            GEMINI_FLASH_LITE_PREVIEW_MODEL: {
                # $0.25 / 1M input tokens, $1.50 / 1M output tokens.
                "input_cost_per_token": 0.00000025,
                "output_cost_per_token": 0.0000015,
            },
        },
        budget_dollars=budget,
        pipeline=levi.PipelineConfig(
            n_llm_workers=8,
            n_eval_processes=8,
            max_tokens=2048,
            eval_timeout=90,
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
        init=levi.InitConfig(
            diversity_model=GEMINI_FLASH_3_MODEL,
        ),
        output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_circle_packing",
    )

    print(f"Best score: {result.best_score:.17g}")
    print(result.best_program)


if __name__ == "__main__":
    main()
