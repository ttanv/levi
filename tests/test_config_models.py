"""Tests for config model invariants and auto-wiring."""

import pytest
from pydantic import ValidationError

from levi.config.models import (
    BudgetConfig,
    LeviConfig,
    SamplerModelPair,
)


def _score_fn(fn, inputs):
    return {"score": sum(fn(x) for x in inputs)}


def _score_fn_no_inputs(fn):
    return {"score": fn(1)}


def _minimal_config_kwargs():
    return {
        "problem_description": "Test problem",
        "function_signature": "def solve(x):",
        "seed_program": "def solve(x):\n    return x",
        "inputs": [1, 2, 3],
        "score_fn": _score_fn,
        "budget": BudgetConfig(dollars=1.0),
    }


class TestSamplerModelPair:
    def test_weight_must_be_positive(self):
        with pytest.raises(ValidationError, match="weight must be positive"):
            SamplerModelPair(sampler="ucb", model="test/model", weight=0.0)


class TestAutoWiring:
    """Tests for the _auto_wire_models model validator."""

    def test_str_paradigm_models_coerced_to_list(self):
        cfg = LeviConfig(**_minimal_config_kwargs(), paradigm_models="openai/gpt-4o")
        assert cfg.paradigm_models == ["openai/gpt-4o"]

    def test_str_mutation_models_coerced_to_list(self):
        cfg = LeviConfig(**_minimal_config_kwargs(), mutation_models="openai/gpt-4o-mini")
        assert cfg.mutation_models == ["openai/gpt-4o-mini"]

    def test_list_models_preserved(self):
        cfg = LeviConfig(
            **_minimal_config_kwargs(),
            paradigm_models=["model-a", "model-b"],
            mutation_models=["model-c", "model-d"],
        )
        assert cfg.paradigm_models == ["model-a", "model-b"]
        assert cfg.mutation_models == ["model-c", "model-d"]

    def test_auto_generates_sampler_model_pairs(self):
        cfg = LeviConfig(
            **_minimal_config_kwargs(),
            mutation_models=["model-a", "model-b"],
        )
        # 2 models × 4 temperatures = 8 pairs
        assert len(cfg.sampler_model_pairs) == 8
        models = {p.model for p in cfg.sampler_model_pairs}
        assert models == {"model-a", "model-b"}
        temps = sorted({p.temperature for p in cfg.sampler_model_pairs})
        assert temps == [0.3, 0.7, 1.0, 1.2]
        assert all(p.sampler == "softmax" for p in cfg.sampler_model_pairs)

    def test_explicit_sampler_model_pairs_not_overridden(self):
        explicit_pairs = [
            SamplerModelPair(sampler="ucb", model="custom/model", weight=2.0),
        ]
        cfg = LeviConfig(
            **_minimal_config_kwargs(),
            sampler_model_pairs=explicit_pairs,
        )
        assert len(cfg.sampler_model_pairs) == 1
        assert cfg.sampler_model_pairs[0].sampler == "ucb"
        assert cfg.sampler_model_pairs[0].model == "custom/model"

    def test_init_diversity_model_auto_filled(self):
        cfg = LeviConfig(
            **_minimal_config_kwargs(),
            paradigm_models=["heavy/model"],
        )
        assert cfg.init.diversity_model == "heavy/model"

    def test_init_variant_models_auto_filled(self):
        cfg = LeviConfig(
            **_minimal_config_kwargs(),
            mutation_models=["light-a", "light-b"],
        )
        assert cfg.init.variant_models == ["light-a", "light-b"]

    def test_meta_advice_model_auto_filled(self):
        cfg = LeviConfig(
            **_minimal_config_kwargs(),
            mutation_models=["light-a", "light-b"],
        )
        assert cfg.meta_advice.model == "light-a"

    def test_pe_heavy_models_auto_filled_from_all_paradigm(self):
        cfg = LeviConfig(
            **_minimal_config_kwargs(),
            paradigm_models=["heavy-a", "heavy-b"],
        )
        assert cfg.punctuated_equilibrium.heavy_models == ["heavy-a", "heavy-b"]

    def test_pe_variant_models_auto_filled(self):
        cfg = LeviConfig(
            **_minimal_config_kwargs(),
            mutation_models=["light-a", "light-b"],
        )
        assert cfg.punctuated_equilibrium.variant_models == ["light-a", "light-b"]

    def test_explicit_sub_config_models_not_overridden(self):
        cfg = LeviConfig(
            **_minimal_config_kwargs(),
            paradigm_models="heavy/model",
            mutation_models="light/model",
            init={"diversity_model": "custom/diversity"},
            meta_advice={"model": "custom/meta"},
            punctuated_equilibrium={"heavy_models": ["custom/heavy"]},
        )
        assert cfg.init.diversity_model == "custom/diversity"
        assert cfg.meta_advice.model == "custom/meta"
        assert cfg.punctuated_equilibrium.heavy_models == ["custom/heavy"]


class TestLeviConfig:
    def test_minimal_config_works_with_defaults(self):
        cfg = LeviConfig(**_minimal_config_kwargs())
        assert cfg.paradigm_models == ["openai/gpt-4o"]
        assert cfg.mutation_models == ["openai/gpt-4o-mini"]
        assert len(cfg.sampler_model_pairs) == 4  # 1 model × 4 temps

    def test_prompt_overrides_preserved(self):
        overrides = {
            "mutation": {"test/model": "Be concise."},
            "paradigm_shift": "Try a fundamentally different algorithm.",
        }
        cfg = LeviConfig(
            **_minimal_config_kwargs(),
            prompt_overrides=overrides,
        )
        assert cfg.prompt_overrides == overrides

    def test_local_endpoints_and_model_info(self):
        cfg = LeviConfig(
            **_minimal_config_kwargs(),
            local_endpoints={"Qwen/Qwen3-30B": "http://localhost:8001/v1"},
            model_info={"Qwen/Qwen3-30B": {"input_cost_per_token": 0.0000001}},
        )
        assert cfg.local_endpoints == {"Qwen/Qwen3-30B": "http://localhost:8001/v1"}
        assert cfg.model_info == {"Qwen/Qwen3-30B": {"input_cost_per_token": 0.0000001}}

    def test_pipeline_max_tokens_default(self):
        cfg = LeviConfig(**_minimal_config_kwargs())
        assert cfg.pipeline.temperature is None
        assert cfg.pipeline.max_tokens == 16384

    def test_provider_sensitive_defaults_are_unset(self):
        cfg = LeviConfig(**_minimal_config_kwargs())
        assert cfg.init.temperature is None
        assert cfg.meta_advice.temperature is None
        assert cfg.punctuated_equilibrium.temperature is None
        assert cfg.punctuated_equilibrium.reasoning_effort is None

    def test_inputs_can_be_omitted_when_score_fn_does_not_require_them(self):
        cfg = LeviConfig(
            problem_description="Test problem",
            function_signature="def solve(x):",
            seed_program="def solve(x):\n    return x",
            score_fn=_score_fn_no_inputs,
            budget=BudgetConfig(dollars=1.0),
        )
        assert cfg.inputs is None
