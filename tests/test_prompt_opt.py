"""Tests for prompt optimization edge cases."""

from levi.config.models import BudgetConfig, LeviConfig
from levi.prompt_opt.optimizer import optimize_prompts


def _score_fn(fn, inputs):
    return {"score": sum(fn(x) for x in inputs)}


def test_prompt_opt_skips_without_seed_program():
    cfg = LeviConfig(
        problem_description="Test problem",
        function_signature="def solve(x):",
        seed_program=None,
        inputs=[1, 2, 3],
        score_fn=_score_fn,
        budget=BudgetConfig(dollars=1.0),
        prompt_opt={"enabled": True},
    )

    overrides, cost = optimize_prompts(cfg)

    assert overrides == {"mutation": {}, "paradigm_shift": None}
    assert cost == 0.0
