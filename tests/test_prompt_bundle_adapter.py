"""Tests for bundle-aware PromptAdapter."""

from levi.artifacts import PromptAdapter
from levi.config import BehaviorConfig, BudgetConfig, LeviConfig, MetaAdviceConfig
from levi.prompts import PromptBundle
from levi.prompts.builder import ProgramWithScore
from levi.core import EvaluationResult, Program


def _make_config(seed_program: str) -> LeviConfig:
    return LeviConfig(
        problem_description="test",
        function_signature="def prompt_artifact():",
        seed_program=seed_program,
        score_fn=lambda x, inputs: {"score": 0.0},
        paradigm_models="openai/gpt-4o",
        mutation_models="openai/gpt-4o-mini",
        budget=BudgetConfig(evaluations=5),
        meta_advice=MetaAdviceConfig(enabled=False),
        behavior=BehaviorConfig(ast_features=[], score_keys=["score"]),
    )


def test_single_prompt_mode_is_not_bundle():
    config = _make_config("hello world")
    adapter = PromptAdapter(config)
    assert adapter.is_bundle_artifact is False

    single_bundle = PromptBundle.single("hello world")
    adapter2 = PromptAdapter(config, seed_bundle=single_bundle)
    assert adapter2.is_bundle_artifact is False


def test_bundle_mode_enabled_for_multiple_targets():
    bundle = PromptBundle.from_mapping({"a": "alpha", "b": "beta"})
    config = _make_config(bundle.serialize())
    adapter = PromptAdapter(config, seed_bundle=bundle)
    assert adapter.is_bundle_artifact is True


def test_bundle_mutation_prompt_targets_component():
    bundle = PromptBundle.from_mapping({"a": "alpha text", "b": "beta text"})
    config = _make_config(bundle.serialize())
    adapter = PromptAdapter(config, seed_bundle=bundle)

    parent = ProgramWithScore(
        Program(content=bundle.serialize(), metadata={}),
        EvaluationResult(scores={"score": 0.5}, is_valid=True),
    )
    prompt = adapter.build_mutation_prompt([parent], target="a")
    assert "[EDITABLE] a" in prompt
    assert "[READ-ONLY] b" in prompt
    assert "Rewrite ONLY the `a`" in prompt


def test_extract_candidate_applies_with_updates_on_target():
    bundle = PromptBundle.from_mapping({"a": "alpha", "b": "beta"})
    config = _make_config(bundle.serialize())
    adapter = PromptAdapter(config, seed_bundle=bundle)

    new_json = adapter.extract_candidate(
        "new alpha text", parent_content=bundle.serialize(), target="a"
    )
    assert new_json is not None
    restored = PromptBundle.from_serialized(new_json)
    assert restored.get("a") == "new alpha text"
    assert restored.get("b") == "beta"


def test_single_prompt_extract_candidate_unchanged():
    config = _make_config("seed")
    adapter = PromptAdapter(config)
    result = adapter.extract_candidate("new prompt text")
    assert result == "new prompt text"


