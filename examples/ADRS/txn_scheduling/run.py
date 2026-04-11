#!/usr/bin/env python3
"""Run Levi for Transaction Scheduling."""

import levi
from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, score_fn


def main() -> None:
    result = levi.evolve_code(
        PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        seed_program=SEED_PROGRAM,
        score_fn=score_fn,
        paradigm_model="openrouter/google/gemini-3-flash-preview",
        mutation_model=[
            "openrouter/xiaomi/mimo-v2-flash",
            levi.LM(
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                api_base="http://localhost:8000/v1",
                api_key="unused",
            ),
        ],
        budget_dollars=13,
        pipeline=levi.PipelineConfig(eval_timeout=300),
    )

    print(f"Best score: {result.best_score:.17g}")


if __name__ == "__main__":
    main()
