#!/usr/bin/env python3
"""Quickstart: evolve a system prompt for sentiment classification.

Six short hand-labelled reviews. LEVI evolves the prompt; the evaluator calls
a small LLM with the prompt against each review and scores accuracy.

    export OPENAI_API_KEY=...
    cd examples/quickstart
    uv run python quickstart_prompts.py

Expected: 2-3 minutes, ~$0.05-0.10 total. Note: the evaluator makes its own
LLM calls, which add to your API bill outside of LEVI's ``budget_dollars``
tracking (which only covers LEVI's mutation/paradigm calls).
"""

import litellm

import levi

MODEL = "openai/gpt-4o-mini"

INPUTS = [
    {"review": "Loved every minute of it, will watch again.", "label": "positive"},
    {"review": "Total waste of two hours. Terrible script.", "label": "negative"},
    {"review": "Solid performances and beautiful cinematography.", "label": "positive"},
    {"review": "Slow, predictable, and forgettable.", "label": "negative"},
    {"review": "Funny, sharp, and emotionally honest throughout.", "label": "positive"},
    {"review": "I wanted to walk out twenty minutes in.", "label": "negative"},
]

SEED_PROMPT = "Classify the review as positive or negative. Reply with one word."

PROBLEM_DESCRIPTION = (
    "Evolve a system prompt that classifies short movie reviews as 'positive' "
    "or 'negative'. The labels are unambiguous; maximize accuracy on the "
    "labelled set."
)


def _classify(prompt: str, review: str) -> str:
    resp = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": review},
        ],
        temperature=0.0,
        max_tokens=4,
    )
    text = (resp.choices[0].message.content or "").lower()
    return "positive" if "positive" in text else "negative"


def score_fn(prompt: str, inputs: list) -> dict:
    correct = 0
    for item in inputs:
        try:
            pred = _classify(prompt, item["review"])
        except Exception:
            return {"score": 0.0}
        if pred == item["label"]:
            correct += 1
    return {"score": 100.0 * correct / len(inputs)}


def main() -> None:
    result = levi.evolve_prompts(
        PROBLEM_DESCRIPTION,
        evaluator=score_fn,
        seed_prompt=SEED_PROMPT,
        inputs=INPUTS,
        model=MODEL,
        budget_dollars=0.10,
    )
    print(f"\nBest accuracy: {result.best_score:.1f}% / 100")
    print(f"Evaluations: {result.total_evaluations}    LEVI cost: ${result.total_cost:.3f}\n")
    print("Best prompt:\n")
    print(result.best_prompt)


if __name__ == "__main__":
    main()
