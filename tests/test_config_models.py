"""Tests for config model invariants and backward compatibility."""

import pytest
from pydantic import ValidationError

from algoforge.config.models import (
    AlgoforgeConfig,
    BudgetConfig,
    LLMProviderConfig,
    SamplerModelPair,
)


def _score_fn(fn, inputs):
    return {"score": sum(fn(x) for x in inputs)}


def _minimal_config_kwargs():
    return {
        "problem_description": "Test problem",
        "function_signature": "def solve(x):",
        "seed_program": "def solve(x):\n    return x",
        "inputs": [1, 2, 3],
        "score_fn": _score_fn,
        "budget": BudgetConfig(dollars=1.0),
        "sampler_model_pairs": [
            SamplerModelPair(sampler="ucb", model="test/model", weight=1.0),
        ],
    }


class TestSamplerModelPair:
    def test_weight_must_be_positive(self):
        with pytest.raises(ValidationError, match="weight must be positive"):
            SamplerModelPair(sampler="ucb", model="test/model", weight=0.0)


class TestAlgoforgeConfig:
    def test_requires_at_least_one_sampler_model_pair(self):
        kwargs = _minimal_config_kwargs()
        kwargs["sampler_model_pairs"] = []

        with pytest.raises(ValidationError, match="must have at least one sampler_model_pair"):
            AlgoforgeConfig(**kwargs)

    def test_migrates_deprecated_api_bases_to_local_endpoints(self):
        cfg = AlgoforgeConfig(
            **_minimal_config_kwargs(),
            api_bases={"local/model": "http://localhost:8001/v1"},
        )

        assert cfg.llm.local_endpoints == {"local/model": "http://localhost:8001/v1"}

    def test_does_not_override_explicit_local_endpoints_during_migration(self):
        cfg = AlgoforgeConfig(
            **_minimal_config_kwargs(),
            api_bases={"legacy/model": "http://legacy:8000/v1"},
            llm=LLMProviderConfig(
                local_endpoints={"local/model": "http://localhost:8001/v1"},
            ),
        )

        assert cfg.llm.local_endpoints == {"local/model": "http://localhost:8001/v1"}

    def test_prompt_overrides_preserved(self):
        overrides = {
            "mutation": {"test/model": "Be concise."},
            "paradigm_shift": "Try a fundamentally different algorithm.",
        }
        cfg = AlgoforgeConfig(
            **_minimal_config_kwargs(),
            prompt_overrides=overrides,
        )

        assert cfg.prompt_overrides == overrides
