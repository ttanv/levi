"""
End-to-end integration tests for levi.evolve_code().

Mock boundary: BaseClient.acompletion. Everything else runs for real.
"""

import threading

import pytest

import levi
from levi.clients import BaseClient, ClientResult

PROGRAMS = [
    "def solve(x):\n    return x * 2",
    "def solve(x):\n    return x ** 2",
    "def solve(x):\n    return x + 1",
    "def solve(x):\n    total = 0\n    for i in range(x):\n        total += i\n    return total",
    "def solve(x):\n    return max(x * 3, x + 10)",
    "def solve(x):\n    if x > 3:\n        return x * x\n    return x * 5",
    "def solve(x):\n    import math\n    return int(math.pow(x, 2))",
    "def solve(x):\n    return sum(range(x))",
]


def _score_fn(fn):
    return {"score": fn(5)}


def _score_fn_with_inputs(fn, inputs):
    return {"score": sum(fn(x) for x in inputs)}


class MockClient(BaseClient):
    """Thread-safe mock that cycles through deterministic code variants."""

    def __init__(self):
        super().__init__("fake/test-model")
        self._counter = 0
        self._lock = threading.Lock()
        self.calls = []

    async def acompletion(self, prompt, **kwargs):
        with self._lock:
            idx = self._counter % len(PROGRAMS)
            self._counter += 1

        code = PROGRAMS[idx]
        content = f"```python\n{code}\n```"

        if isinstance(prompt, list):
            prompt_text = prompt[0]["content"] if prompt else ""
        else:
            prompt_text = prompt

        self.calls.append(
            {
                "model": self.model,
                "prompt_snippet": prompt_text[:200],
            }
        )

        return ClientResult(text=content, cost=0.001)


@pytest.mark.slow
class TestFullRun:
    def test_basic_evolution(self):
        mock_client = MockClient()

        result = levi.evolve_code(
            "Maximize the return value of solve(5)",
            function_signature="def solve(x):",
            seed_program="def solve(x):\n    return x",
            score_fn=_score_fn,
            model=mock_client,
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

        assert result.best_score >= 25
        assert result.total_evaluations >= 5
        assert result.archive_size >= 2

        ns = {}
        exec(result.best_program, ns)
        actual_score = ns["solve"](5)
        assert actual_score >= result.best_score

        assert result.total_cost > 0
        assert 0 < result.runtime_seconds < 60
        assert len(mock_client.calls) >= 4

    def test_with_cascade(self):
        mock_client = MockClient()

        result = levi.evolve_code(
            "Maximize the sum of solve(x) over inputs",
            function_signature="def solve(x):",
            seed_program="def solve(x):\n    return x",
            score_fn=_score_fn_with_inputs,
            inputs=[3, 5, 7],
            model=mock_client,
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

        assert result.best_score >= 50
        assert result.total_evaluations >= 5
        assert result.best_program
        assert result.archive_size >= 1

    def test_no_seed_program(self):
        mock_client = MockClient()

        result = levi.evolve_code(
            "Maximize solve(5)",
            function_signature="def solve(x):",
            score_fn=_score_fn,
            model=mock_client,
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

        assert result.best_score >= 6
        assert result.total_evaluations >= 3
        assert result.best_program
