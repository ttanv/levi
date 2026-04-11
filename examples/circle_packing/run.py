#!/usr/bin/env python3
"""Run Levi on n=26 circle packing."""

import levi
from problem import FUNCTION_SIGNATURE, PROBLEM_DESCRIPTION, score_fn


def main() -> None:
    result = levi.evolve_code(
        PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        score_fn=score_fn,
        paradigm_model="openrouter/google/gemini-3-flash-preview",
        mutation_model=[
            levi.Client(
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                api_base="http://localhost:8000/v1",
                api_key="unused",
            )
        ],
        budget_dollars=1,
        pipeline=levi.PipelineConfig(
            n_llm_workers=8,
            n_eval_processes=8,
            eval_timeout=600,
        ),
    )

    print(f"Best score: {result.best_score:.17g}")
    print(result.best_program)


if __name__ == "__main__":
    main()
