#!/usr/bin/env python3
"""Quickstart: evolve a bin-packing function via your Claude Code subscription.

No API key, no $ spend — uses the local ``claude`` CLI in print mode with all
tools disabled. The CLI must be installed and signed in:
    https://claude.com/code

    cd examples/quickstart
    uv run python quickstart_claude.py

Expected: a couple of minutes for 20 evaluations.
"""

import levi
from problem import FUNCTION_SIGNATURE, ITEMS_LIST, PROBLEM_DESCRIPTION, SEED_PROGRAM, score_fn


def main() -> None:
    haiku = levi.ClaudeCodeClient(model="haiku")

    result = levi.evolve_code(
        PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        seed_program=SEED_PROGRAM,
        score_fn=score_fn,
        inputs=ITEMS_LIST,
        model=haiku,
        budget_evals=20,
        init=levi.InitConfig(n_diverse_seeds=2, n_variants_per_seed=3),
        punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(enabled=False),
        pipeline=levi.PipelineConfig(n_llm_workers=2, n_eval_processes=2),
    )
    print(f"\nBest score: {result.best_score:.3f} / 100")
    print(f"Evaluations: {result.total_evaluations}\n")
    print("Best program:\n")
    print(result.best_program)


if __name__ == "__main__":
    main()
