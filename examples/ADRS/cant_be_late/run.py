#!/usr/bin/env python3
"""Run Levi for Can't Be Late Scheduling."""

from datetime import datetime

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, INPUTS, score_fn
import levi


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
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
        ],
        budget_dollars=4.50,
        local_endpoints={"Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8000/v1"},
        pipeline=levi.PipelineConfig(n_llm_workers=12, n_eval_processes=12, n_inspirations=1, eval_timeout=300),
        behavior=levi.BehaviorConfig(
            ast_features=['cyclomatic_complexity', 'comparison_count', 'math_operators', 'branch_count'],
            score_keys=['tight_deadline_score', 'loose_deadline_score'],
        ),
        prompt_opt=levi.PromptOptConfig(enabled=True),
        output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_po",
    )

    print(f"Best score: {result.best_score:.17g}")


if __name__ == "__main__":
    main()
