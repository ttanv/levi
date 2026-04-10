"""
End-to-end integration tests for levi.evolve_code().

Mock boundary: UnifiedLLMClient.acompletion — everything else runs for real
(code extraction, exec(), scoring, behavior extraction, CVT pool, producer-consumer,
punctuated equilibrium, budget tracking).

These tests serve as regression gates for large refactors.
"""

import threading
from unittest.mock import patch

import pytest

import levi
from levi.llm import CompletionResponse, UnifiedLLMClient

# ---------------------------------------------------------------------------
# Deterministic program variants with known scores for score_fn = fn(5)
# ---------------------------------------------------------------------------

PROGRAMS = [
    # score=10, no loops/branches/math_ops
    "def solve(x):\n    return x * 2",
    # score=25, has math operator (**)
    "def solve(x):\n    return x ** 2",
    # score=6, minimal
    "def solve(x):\n    return x + 1",
    # score=10, has loop — different behavior vector
    "def solve(x):\n    total = 0\n    for i in range(x):\n        total += i\n    return total",
    # score=15, has call (max)
    "def solve(x):\n    return max(x * 3, x + 10)",
    # score=25, has branch — different behavior vector
    "def solve(x):\n    if x > 3:\n        return x * x\n    return x * 5",
    # score=25, has call + import
    "def solve(x):\n    import math\n    return int(math.pow(x, 2))",
    # score=10, has call (sum, range)
    "def solve(x):\n    return sum(range(x))",
]


def _score_fn(fn):
    """Trivial scorer: just call fn(5)."""
    return {"score": fn(5)}


def _score_fn_with_inputs(fn, inputs):
    """Scorer that uses inputs list."""
    return {"score": sum(fn(x) for x in inputs)}


class MockLLM:
    """Thread-safe mock that cycles through PROGRAMS, returning them as code blocks."""

    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()
        self.calls = []

    async def acompletion(self, *, model, messages, **kwargs):
        with self._lock:
            idx = self._counter % len(PROGRAMS)
            self._counter += 1

        code = PROGRAMS[idx]
        content = f"```python\n{code}\n```"

        self.calls.append({
            "model": model,
            "prompt_snippet": messages[0]["content"][:200] if messages else "",
        })

        return CompletionResponse(
            content=content,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model=model,
            cost=0.001,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFullRun:
    """Integration tests that exercise the full evolve_code pipeline."""

    @patch("litellm.register_model")
    def test_basic_evolution(self, _mock_register):
        """Full pipeline: init, evolution loop, PE, archive, result."""
        mock_llm = MockLLM()

        with patch.object(UnifiedLLMClient, "acompletion", side_effect=mock_llm.acompletion):
            result = levi.evolve_code(
                "Maximize the return value of solve(5)",
                function_signature="def solve(x):",
                seed_program="def solve(x):\n    return x",
                score_fn=_score_fn,
                model="fake/test-model",
                budget_evals=15,
                init=levi.InitConfig(n_diverse_seeds=2, n_variants_per_seed=2),
                pipeline=levi.PipelineConfig(
                    n_llm_workers=2,
                    n_eval_processes=2,
                    eval_timeout=5.0,
                ),
                punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(
                    enabled=True,
                    interval=5,
                    n_clusters=2,
                    n_variants=1,
                ),
                cascade=levi.CascadeConfig(enabled=False),
                meta_advice=levi.MetaAdviceConfig(enabled=False),
                cvt=levi.CVTConfig(n_centroids=10, data_driven_centroids=True),
                behavior=levi.BehaviorConfig(
                    ast_features=["loop_count", "branch_count", "math_operators"],
                ),
            )

        # The x**2 program scores 25 — must be found
        assert result.best_score >= 25, f"Expected best_score >= 25, got {result.best_score}"

        # Pipeline actually ran
        assert result.total_evaluations >= 5, f"Expected >= 5 evals, got {result.total_evaluations}"
        assert result.archive_size >= 2, f"Expected archive_size >= 2, got {result.archive_size}"

        # Best program is valid and produces the claimed score
        assert result.best_program, "best_program is empty"
        ns = {}
        exec(result.best_program, ns)
        actual_score = ns["solve"](5)
        assert actual_score >= result.best_score, (
            f"best_program produces {actual_score} but best_score is {result.best_score}"
        )

        # Cost and runtime tracking
        assert result.total_cost > 0, "total_cost should be > 0"
        assert 0 < result.runtime_seconds < 60, f"runtime_seconds={result.runtime_seconds}"

        # LLM was actually called (init + evolution + PE)
        assert len(mock_llm.calls) >= 4, f"Expected >= 4 LLM calls, got {len(mock_llm.calls)}"

    @patch("litellm.register_model")
    def test_with_cascade(self, _mock_register):
        """Full pipeline with cascade evaluation enabled."""
        mock_llm = MockLLM()

        with patch.object(UnifiedLLMClient, "acompletion", side_effect=mock_llm.acompletion):
            result = levi.evolve_code(
                "Maximize the sum of solve(x) over inputs",
                function_signature="def solve(x):",
                seed_program="def solve(x):\n    return x",
                score_fn=_score_fn_with_inputs,
                inputs=[3, 5, 7],
                model="fake/test-model",
                budget_evals=15,
                init=levi.InitConfig(n_diverse_seeds=2, n_variants_per_seed=2),
                pipeline=levi.PipelineConfig(
                    n_llm_workers=2,
                    n_eval_processes=2,
                    eval_timeout=5.0,
                ),
                punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(enabled=False),
                cascade=levi.CascadeConfig(
                    enabled=True,
                    quick_inputs=[3],
                    min_score_ratio=0.5,
                    quick_timeout=5.0,
                ),
                meta_advice=levi.MetaAdviceConfig(enabled=False),
                cvt=levi.CVTConfig(n_centroids=10, data_driven_centroids=True),
                behavior=levi.BehaviorConfig(
                    ast_features=["loop_count", "branch_count", "math_operators"],
                ),
            )

        # x**2 scores 9+25+49=83 over inputs [3,5,7]
        assert result.best_score >= 50, f"Expected best_score >= 50, got {result.best_score}"
        assert result.total_evaluations >= 5
        assert result.best_program, "best_program is empty"
        assert result.archive_size >= 1

    @patch("litellm.register_model")
    def test_no_seed_program(self, _mock_register):
        """Pipeline works without a seed program (init generates from scratch)."""
        mock_llm = MockLLM()

        with patch.object(UnifiedLLMClient, "acompletion", side_effect=mock_llm.acompletion):
            result = levi.evolve_code(
                "Maximize solve(5)",
                function_signature="def solve(x):",
                score_fn=_score_fn,
                model="fake/test-model",
                budget_evals=12,
                init=levi.InitConfig(n_diverse_seeds=2, n_variants_per_seed=1),
                pipeline=levi.PipelineConfig(
                    n_llm_workers=1,
                    n_eval_processes=2,
                    eval_timeout=5.0,
                ),
                punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(enabled=False),
                cascade=levi.CascadeConfig(enabled=False),
                meta_advice=levi.MetaAdviceConfig(enabled=False),
                cvt=levi.CVTConfig(n_centroids=8, data_driven_centroids=True),
                behavior=levi.BehaviorConfig(
                    ast_features=["loop_count", "branch_count", "math_operators"],
                ),
            )

        assert result.best_score >= 6, f"Expected best_score >= 6, got {result.best_score}"
        assert result.total_evaluations >= 3
        assert result.best_program
