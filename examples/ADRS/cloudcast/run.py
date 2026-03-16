#!/usr/bin/env python3
"""Run Levi for Cloudcast Broadcast Optimization."""

from datetime import datetime

import levi
from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, score_fn


def main() -> None:
    result = levi.evolve_code(
        PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        seed_program=SEED_PROGRAM,
        score_fn=score_fn,
        paradigm_model="openrouter/google/gemini-3-flash-preview",
        mutation_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        local_endpoints={"Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8000/v1"},
        budget_dollars=3.00,
        behavior=levi.BehaviorConfig(
            ast_features=['loop_count', 'branch_count', 'math_operators'],
            score_keys=[
                'intra_aws_score',
                'intra_azure_score',
                'intra_gcp_score',
                'inter_agz_score',
                'inter_gaz2_score',
            ],
        ),
        pipeline=levi.PipelineConfig(
            n_llm_workers=8,
            n_eval_processes=8,
            eval_timeout=120,
        ),
        output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_cloudcast",
    )

    print(f"Best score: {result.best_score:.17g}")


if __name__ == "__main__":
    main()
