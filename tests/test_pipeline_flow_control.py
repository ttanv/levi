"""Tests for pipeline flow-control behavior (queue bounds, stall stop, empty archive stop)."""

import asyncio

import levi.pipeline.runner as runner_module
from levi.config.models import BudgetConfig, LeviConfig
from levi.pipeline.producer import llm_producer
from levi.pipeline.runner import PipelineRunner
from levi.pipeline.state import PipelineState


def _score_fn(fn, inputs):
    return {"score": sum(fn(x) for x in inputs)}


def _make_config(*, pipeline: dict | None = None) -> LeviConfig:
    kwargs = {
        "problem_description": "Test problem",
        "function_signature": "def solve(x):",
        "seed_program": "def solve(x):\n    return x",
        "inputs": [1, 2, 3],
        "score_fn": _score_fn,
        "budget": BudgetConfig(dollars=100.0),
    }
    if pipeline is not None:
        kwargs["pipeline"] = pipeline
    return LeviConfig(**kwargs)


class _DummyPool:
    def get_stats(self):
        return {"best_score": 0.0}

    def size(self):
        return 0


class _EmptyPool:
    def size(self):
        return 0


class _DummyExecutor:
    pass


class TestPipelineRunnerFlowControl:
    def test_runner_auto_sizes_code_queue(self):
        config = _make_config(pipeline={"n_eval_processes": 3})
        runner = PipelineRunner(config, _DummyPool(), _DummyExecutor())

        assert runner.code_queue.maxsize == 6

    def test_wait_for_completion_stops_on_idle_stall(self, monkeypatch):
        monkeypatch.setattr(runner_module, "_MIN_STALL_TIMEOUT_SECONDS", 0.1)
        config = _make_config(
            pipeline={
                "n_llm_workers": 1,
                "n_eval_processes": 1,
                "eval_timeout": 0.01,
            }
        )
        state = PipelineState(config.budget)
        runner = PipelineRunner(config, _DummyPool(), _DummyExecutor(), state=state)

        asyncio.run(runner._wait_for_completion())

        assert runner.stop_event.is_set() is True
        assert state.budget_exhausted is False


class TestProducerFlowControl:
    def test_producer_stops_when_archive_empty(self):
        config = _make_config()
        state = PipelineState(config.budget)
        stop_event = asyncio.Event()
        code_queue = asyncio.Queue()
        archive_lock = asyncio.Lock()

        asyncio.run(
            llm_producer(
                worker_id=0,
                code_queue=code_queue,
                pool=_EmptyPool(),
                archive_lock=archive_lock,
                config=config,
                state=state,
                stop_event=stop_event,
            )
        )

        assert stop_event.is_set() is True
        assert code_queue.empty() is True
