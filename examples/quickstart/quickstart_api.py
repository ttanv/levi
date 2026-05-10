#!/usr/bin/env python3
"""Quickstart: evolve a bin-packing function via an API model.

Default uses OpenAI's `gpt-4o-mini`. Switch `MODEL` to any other litellm
provider id (e.g. ``anthropic/claude-haiku-4-5``, ``openrouter/google/gemini-2.5-flash``)
and set the corresponding API key in your environment.

    export OPENAI_API_KEY=...
    cd examples/quickstart
    uv run python quickstart_api.py

Expected: 1-2 minutes, ~$0.05-0.10 total spend.
"""

import levi
from problem import FUNCTION_SIGNATURE, ITEMS_LIST, PROBLEM_DESCRIPTION, SEED_PROGRAM, score_fn

MODEL = "openai/gpt-4o-mini"


def main() -> None:
    result = levi.evolve_code(
        PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        seed_program=SEED_PROGRAM,
        score_fn=score_fn,
        inputs=ITEMS_LIST,
        model=MODEL,
        budget_dollars=0.10,
    )
    print(f"\nBest score: {result.best_score:.3f} / 100")
    print(f"Evaluations: {result.total_evaluations}    Cost: ${result.total_cost:.3f}\n")
    print("Best program:\n")
    print(result.best_program)


if __name__ == "__main__":
    main()
