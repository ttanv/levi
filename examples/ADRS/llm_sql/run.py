#!/usr/bin/env python3
"""Run Levi for LLM SQL (Column Reordering)."""

from datetime import datetime

import levi
from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, INPUTS, score_fn


def main() -> None:
    result = levi.evolve_code(
        PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        seed_program=SEED_PROGRAM,
        score_fn=score_fn,
        inputs=INPUTS,
        paradigm_model="openrouter/google/gemini-3-flash-preview",
        mutation_model=[
            "openrouter/xiaomi/mimo-v2-flash",
            levi.Client(
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                api_base="http://localhost:8000/v1",
                api_key="unused",
                input_cost_per_token=0.0000001,
                output_cost_per_token=0.0000004,
            ),
        ],
        budget_dollars=4.50,
        behavior=levi.BehaviorConfig(
            ast_features=['loop_nesting_max', 'comparison_count', 'call_count', 'branch_count'],
        ),
        output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_po",
    )

    print(f"Best score: {result.best_score:.17g}")


if __name__ == "__main__":
    main()
