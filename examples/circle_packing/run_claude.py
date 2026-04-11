#!/usr/bin/env python3
"""Run Levi on n=26 circle packing using a Claude Code subscription (haiku).

Tiny test run: a handful of seeds/variants and a 10-eval budget cap. Requires
the ``claude`` CLI to be installed locally and signed into a Claude
subscription.
"""

import levi
from problem import FUNCTION_SIGNATURE, PROBLEM_DESCRIPTION, score_fn


def main() -> None:
    haiku = levi.ClaudeCodeClient(model="haiku")

    result = levi.evolve_code(
        PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        score_fn=score_fn,
        paradigm_model=haiku,
        mutation_model=haiku,
        budget_evals=10,
        init=levi.InitConfig(n_diverse_seeds=2, n_variants_per_seed=3),
        punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(enabled=False),
        pipeline=levi.PipelineConfig(
            n_llm_workers=2,
            n_eval_processes=2,
            eval_timeout=600,
        ),
    )

    print(f"Best score: {result.best_score:.17g}")
    print(result.best_program)


if __name__ == "__main__":
    main()
