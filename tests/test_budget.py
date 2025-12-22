"""
Tests for budget module: BudgetManager and BudgetExhausted.

These enforce resource constraints across the optimization run.
"""

import pytest
import time

from algoforge.budget import BudgetManager, BudgetExhausted, ResourceType


class TestResourceType:
    """Tests for ResourceType enum."""

    def test_resource_types_exist(self):
        """All expected resource types are defined."""
        assert ResourceType.LLM_TOKENS.value == "llm_tokens"
        assert ResourceType.LLM_COST.value == "llm_cost"
        assert ResourceType.EVALUATIONS.value == "evaluations"
        assert ResourceType.WALL_TIME.value == "wall_time"


class TestBudgetManager:
    """Tests for BudgetManager."""

    def test_creation_with_no_limits(self):
        """BudgetManager can be created with no limits (unlimited)."""
        budget = BudgetManager()

        assert budget.max_llm_tokens is None
        assert budget.max_llm_cost is None
        assert budget.max_evaluations is None
        assert budget.max_wall_time is None

    def test_creation_with_all_limits(self):
        """BudgetManager can be created with all limits."""
        budget = BudgetManager(
            max_llm_tokens=100000,
            max_llm_cost=10.0,
            max_evaluations=500,
            max_wall_time=3600.0,
        )

        assert budget.max_llm_tokens == 100000
        assert budget.max_llm_cost == 10.0
        assert budget.max_evaluations == 500
        assert budget.max_wall_time == 3600.0

    def test_consume_tokens_success(self):
        """consume() returns True when within budget."""
        budget = BudgetManager(max_llm_tokens=1000)

        result = budget.consume(ResourceType.LLM_TOKENS, 500)

        assert result is True
        assert budget.remaining(ResourceType.LLM_TOKENS) == 500

    def test_consume_tokens_failure(self):
        """consume() returns False when exceeds budget."""
        budget = BudgetManager(max_llm_tokens=1000)

        result = budget.consume(ResourceType.LLM_TOKENS, 1500)

        assert result is False
        # Should not have consumed anything
        assert budget.remaining(ResourceType.LLM_TOKENS) == 1000

    def test_consume_cost_success(self):
        """consume() works for LLM_COST."""
        budget = BudgetManager(max_llm_cost=5.0)

        result = budget.consume(ResourceType.LLM_COST, 2.5)

        assert result is True
        assert budget.remaining(ResourceType.LLM_COST) == 2.5

    def test_consume_evaluations_success(self):
        """consume() works for EVALUATIONS."""
        budget = BudgetManager(max_evaluations=100)

        result = budget.consume(ResourceType.EVALUATIONS, 1)

        assert result is True
        assert budget.remaining(ResourceType.EVALUATIONS) == 99

    def test_consume_multiple_times(self):
        """Multiple consume calls accumulate."""
        budget = BudgetManager(max_llm_tokens=1000)

        budget.consume(ResourceType.LLM_TOKENS, 300)
        budget.consume(ResourceType.LLM_TOKENS, 200)

        assert budget.remaining(ResourceType.LLM_TOKENS) == 500

    def test_try_consume_success(self):
        """try_consume() doesn't raise when within budget."""
        budget = BudgetManager(max_evaluations=10)

        # Should not raise
        budget.try_consume(ResourceType.EVALUATIONS, 1)

        assert budget.remaining(ResourceType.EVALUATIONS) == 9

    def test_try_consume_raises_on_exceed(self):
        """try_consume() raises BudgetExhausted when exceeds limit."""
        budget = BudgetManager(max_evaluations=10)

        with pytest.raises(BudgetExhausted) as exc_info:
            budget.try_consume(ResourceType.EVALUATIONS, 15)

        assert exc_info.value.resource_type == ResourceType.EVALUATIONS
        assert exc_info.value.remaining == 10
        assert exc_info.value.requested == 15

    def test_remaining_unlimited(self):
        """remaining() returns None for unlimited resources."""
        budget = BudgetManager()  # No limits

        assert budget.remaining(ResourceType.LLM_TOKENS) is None
        assert budget.remaining(ResourceType.LLM_COST) is None
        assert budget.remaining(ResourceType.EVALUATIONS) is None
        assert budget.remaining(ResourceType.WALL_TIME) is None

    def test_remaining_wall_time(self):
        """remaining() correctly calculates wall time remaining."""
        budget = BudgetManager(max_wall_time=10.0)

        # Immediately after creation, should be close to 10
        remaining = budget.remaining(ResourceType.WALL_TIME)
        assert remaining is not None
        assert 9.9 <= remaining <= 10.0

    def test_is_exhausted_false_initially(self):
        """is_exhausted() returns False when budgets have room."""
        budget = BudgetManager(max_evaluations=100, max_llm_cost=10.0)

        assert budget.is_exhausted() is False

    def test_is_exhausted_true_when_limit_reached(self):
        """is_exhausted() returns True when any limit is reached."""
        budget = BudgetManager(max_evaluations=5)

        for _ in range(5):
            budget.consume(ResourceType.EVALUATIONS, 1)

        assert budget.is_exhausted() is True

    def test_is_exhausted_unlimited(self):
        """is_exhausted() returns False when no limits set."""
        budget = BudgetManager()  # No limits

        assert budget.is_exhausted() is False

    def test_check_budget_success(self):
        """check_budget() doesn't raise when budget available."""
        budget = BudgetManager(max_evaluations=100)

        # Should not raise
        budget.check_budget()

    def test_check_budget_raises_when_exhausted(self):
        """check_budget() raises when budget exhausted."""
        budget = BudgetManager(max_evaluations=0)

        with pytest.raises(BudgetExhausted) as exc_info:
            budget.check_budget()

        assert exc_info.value.resource_type == ResourceType.EVALUATIONS

    def test_elapsed_time(self):
        """elapsed_time property tracks time since creation."""
        budget = BudgetManager()

        # Should be close to 0 immediately
        assert budget.elapsed_time < 0.1

        time.sleep(0.1)

        # Should have elapsed about 0.1s
        assert 0.09 <= budget.elapsed_time <= 0.2

    def test_wall_time_exhaustion(self):
        """Wall time limit correctly exhausts budget."""
        budget = BudgetManager(max_wall_time=0.1)

        assert budget.is_exhausted() is False

        time.sleep(0.15)

        assert budget.is_exhausted() is True

    def test_thread_safety(self):
        """BudgetManager operations are thread-safe."""
        import threading

        budget = BudgetManager(max_evaluations=1000)
        results = []

        def consume_many():
            for _ in range(100):
                result = budget.consume(ResourceType.EVALUATIONS, 1)
                results.append(result)

        threads = [threading.Thread(target=consume_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed (1000 available, 1000 consumed)
        assert all(results)
        assert budget.remaining(ResourceType.EVALUATIONS) == 0


class TestBudgetExhausted:
    """Tests for BudgetExhausted exception."""

    def test_exception_attributes(self):
        """BudgetExhausted stores correct attributes."""
        exc = BudgetExhausted(
            resource_type=ResourceType.LLM_COST,
            remaining=2.5,
            requested=5.0,
        )

        assert exc.resource_type == ResourceType.LLM_COST
        assert exc.remaining == 2.5
        assert exc.requested == 5.0

    def test_str_with_requested(self):
        """String representation includes requested amount."""
        exc = BudgetExhausted(
            resource_type=ResourceType.LLM_COST,
            remaining=2.5,
            requested=5.0,
        )

        msg = str(exc)
        assert "5.0" in msg
        assert "2.5" in msg
        assert "llm_cost" in msg

    def test_str_without_requested(self):
        """String representation works without requested amount."""
        exc = BudgetExhausted(
            resource_type=ResourceType.EVALUATIONS,
            remaining=0,
        )

        msg = str(exc)
        assert "evaluations" in msg
        assert "remaining: 0" in msg

    def test_is_exception(self):
        """BudgetExhausted is a proper Exception."""
        exc = BudgetExhausted(
            resource_type=ResourceType.LLM_TOKENS,
            remaining=0,
        )

        assert isinstance(exc, Exception)

        # Can be raised and caught
        with pytest.raises(BudgetExhausted):
            raise exc
