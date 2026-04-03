"""Tests for per-cell cascade behavior in the eval consumer."""

import asyncio

from levi.config.models import BudgetConfig, CascadeConfig, LeviConfig, PipelineConfig
from levi.core import EvaluationResult
from levi.pipeline.consumer import eval_consumer
from levi.pipeline.state import PipelineState


def _score_fn(_fn, _inputs):
    return {"score": 1.0}


def _make_config() -> LeviConfig:
    return LeviConfig(
        problem_description="Test problem",
        function_signature="def solve(x):",
        inputs=[1],
        score_fn=_score_fn,
        budget=BudgetConfig(evaluations=10),
        pipeline=PipelineConfig(n_llm_workers=1, n_eval_processes=1, eval_timeout=5.0),
        cascade=CascadeConfig(enabled=True, quick_inputs=["quick"], min_score_ratio=1.0, quick_timeout=1.0),
    )


class _Executor:
    def __init__(self, *results: dict):
        self.results = list(results)
        self.calls = 0

    async def run(self, *_args, **_kwargs):
        self.calls += 1
        if not self.results:
            raise AssertionError("executor.run called more times than expected")
        return dict(self.results.pop(0))


class _Elite:
    def __init__(self, scores: dict):
        self.result = EvaluationResult(scores=scores, is_valid=True)


class _Pool:
    def __init__(self, *, preview_cell: int = 0, incumbent_scores: dict | None = None):
        self.preview_cell_value = preview_cell
        self.incumbent = _Elite(incumbent_scores) if incumbent_scores is not None else None
        self.add_calls = []
        self.update_calls = []

    def preview_cell(self, _program, _scores=None):
        return self.preview_cell_value

    def get_elite(self, cell_index: int):
        assert cell_index == self.preview_cell_value
        return self.incumbent

    def add(self, program, eval_result):
        self.add_calls.append((program, eval_result))
        return True, self.preview_cell_value

    def update_sampler(self, sampler_name: str, cell: int, success: bool):
        self.update_calls.append((sampler_name, cell, success))

    def size(self):
        return 1 if self.add_calls else 0


async def _run_consumer_once(pool: _Pool, executor: _Executor, config: LeviConfig) -> PipelineState:
    state = PipelineState(config.budget)
    queue = asyncio.Queue()
    await queue.put({"code": "def solve(x):\n    return x", "sampler": "softmax", "source_cell": 3, "model": "m"})
    stop_event = asyncio.Event()
    stop_event.set()
    archive_lock = asyncio.Lock()

    await eval_consumer(
        worker_id=0,
        code_queue=queue,
        pool=pool,
        archive_lock=archive_lock,
        executor=executor,
        config=config,
        state=state,
        stop_event=stop_event,
    )
    return state


class TestEvalConsumerPerCellCascade:
    def test_runs_full_eval_for_empty_target_cell(self):
        config = _make_config()
        pool = _Pool()
        executor = _Executor({"score": 0.6}, {"score": 1.2})

        state = asyncio.run(_run_consumer_once(pool, executor, config))

        assert executor.calls == 2
        assert len(pool.add_calls) == 1
        assert pool.add_calls[0][1].scores["quick_score"] == 0.6
        assert state.eval_count == 1

    def test_skips_full_eval_when_quick_score_does_not_beat_cell_incumbent(self):
        config = _make_config()
        pool = _Pool(incumbent_scores={"score": 5.0, "quick_score": 0.8})
        executor = _Executor({"score": 0.7})

        state = asyncio.run(_run_consumer_once(pool, executor, config))

        assert executor.calls == 1
        assert pool.add_calls == []
        assert pool.update_calls == [("softmax", 3, False)]
        assert state.eval_count == 1

    def test_runs_full_eval_when_quick_score_beats_cell_incumbent(self):
        config = _make_config()
        pool = _Pool(incumbent_scores={"score": 5.0, "quick_score": 0.8})
        executor = _Executor({"score": 0.9}, {"score": 1.5})

        state = asyncio.run(_run_consumer_once(pool, executor, config))

        assert executor.calls == 2
        assert len(pool.add_calls) == 1
        assert pool.add_calls[0][1].scores["score"] == 1.5
        assert pool.add_calls[0][1].scores["quick_score"] == 0.9
        assert pool.update_calls == [("softmax", 3, True)]
        assert state.eval_count == 1

    def test_runs_full_eval_when_cell_incumbent_has_no_quick_score(self):
        config = _make_config()
        pool = _Pool(incumbent_scores={"score": 5.0})
        executor = _Executor({"score": 0.5}, {"score": 1.1})

        state = asyncio.run(_run_consumer_once(pool, executor, config))

        assert executor.calls == 2
        assert len(pool.add_calls) == 1
        assert pool.add_calls[0][1].scores["quick_score"] == 0.5
        assert state.eval_count == 1
