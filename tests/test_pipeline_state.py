"""Tests for pipeline state bookkeeping and budget semantics."""

import math

from algoforge.config.models import BudgetConfig
from algoforge.pipeline.state import PipelineState


class TestPipelineStateBudget:
    def test_budget_exhausted_by_dollars(self):
        state = PipelineState(BudgetConfig(dollars=10.0))
        state.total_cost = 10.0
        assert state.budget_exhausted is True

    def test_budget_exhausted_by_evaluations(self):
        state = PipelineState(BudgetConfig(evaluations=5))
        state.eval_count = 5
        assert state.budget_exhausted is True

    def test_budget_exhausted_by_seconds(self, monkeypatch):
        state = PipelineState(BudgetConfig(seconds=10.0), start_time=100.0)
        monkeypatch.setattr("algoforge.pipeline.state.time.time", lambda: 111.0)
        assert state.budget_exhausted is True

    def test_budget_progress_uses_primary_limit_order(self, monkeypatch):
        state = PipelineState(
            BudgetConfig(dollars=20.0, evaluations=100, seconds=200.0),
            start_time=100.0,
        )
        state.total_cost = 5.0
        state.eval_count = 90
        monkeypatch.setattr("algoforge.pipeline.state.time.time", lambda: 280.0)

        # Dollars take precedence over evaluations and time.
        assert state.budget_progress == 0.25

    def test_budget_handles_zero_limits_without_crashing(self):
        state = PipelineState(BudgetConfig(dollars=0.0, evaluations=0, seconds=0.0))
        assert state.budget_exhausted is True
        assert state.budget_progress == 1.0

    def test_elapsed_seconds(self, monkeypatch):
        state = PipelineState(BudgetConfig(), start_time=10.0)
        monkeypatch.setattr("algoforge.pipeline.state.time.time", lambda: 13.5)
        assert state.elapsed_seconds == 3.5


class TestPipelineStateMetrics:
    def test_record_accept_reject_error(self):
        state = PipelineState(BudgetConfig())
        long_error = "x" * 150

        state.record_accept()
        state.record_reject()
        state.record_error(long_error)

        assert state.eval_count == 3
        assert state.accept_count == 1
        assert state.error_count == 1
        assert state.period_acceptances == 1
        assert state.period_rejections == 1
        assert state.period_errors == 1
        assert len(next(iter(state.period_error_messages))) == 100
        assert state.all_error_counts[long_error[:100]] == 1

    def test_should_generate_meta_advice_interval(self):
        state = PipelineState(BudgetConfig())
        state.eval_count = 50

        assert state.should_generate_meta_advice(interval=50) is True

        state.meta_advice_eval_count = 50
        assert state.should_generate_meta_advice(interval=50) is False
        assert state.should_generate_meta_advice(interval=0) is False

    def test_reset_period_metrics_returns_snapshot_and_clears(self):
        state = PipelineState(BudgetConfig())
        state.record_accept()
        state.record_reject()
        state.record_error("TypeError: bad input")
        state.record_error("TypeError: bad input")

        metrics = state.reset_period_metrics()

        assert metrics["acceptances"] == 1
        assert metrics["rejections"] == 1
        assert metrics["errors"] == 2
        assert ("TypeError: bad input", 2) in metrics["top_errors"]
        assert state.period_acceptances == 0
        assert state.period_rejections == 0
        assert state.period_errors == 0
        assert state.period_error_messages == set()
        assert state.meta_advice_eval_count == state.eval_count

    def test_record_score_tracks_running_best(self, monkeypatch):
        state = PipelineState(BudgetConfig(), start_time=100.0)
        monkeypatch.setattr("algoforge.pipeline.state.time.time", lambda: 130.0)

        state.eval_count = 1
        state.total_cost = 0.5
        state.record_score(10.0, accepted=True, sampler="ucb", archive_size=1, cell_index=7)

        state.eval_count = 2
        state.total_cost = 0.9
        state.record_score(9.5, accepted=False, sampler="uniform", archive_size=1, cell_index=7)

        assert state.best_score_so_far == 10.0
        assert len(state.score_history) == 2
        assert state.score_history[0].best_score == 10.0
        assert state.score_history[1].best_score == 10.0
        assert state.get_score_history_list() == [10.0, 10.0]

    def test_add_cost_ignores_invalid_values(self):
        state = PipelineState(BudgetConfig())

        state.add_cost(1.0)
        state.add_cost(float("inf"))
        state.add_cost(float("nan"))
        state.add_cost(-10.0)
        state.add_cost("invalid")

        assert math.isclose(state.total_cost, 1.0)
