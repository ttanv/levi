"""
Tests for pool module: CVTMAPElitesPool.

Tests the CVT-MAP-Elites archive with multi-strategy sampling.
"""

import pytest
import random
import numpy as np

from algoforge.core import Program, EvaluationResult
from algoforge.pool import CVTMAPElitesPool, SampleResult
from algoforge.behavior import BehaviorExtractor


class TestSampleResult:
    """Tests for SampleResult dataclass."""

    def test_sample_result_creation(self):
        """SampleResult can be created with parent and inspirations."""
        parent = Program(code="def solve(x): return x")
        insp1 = Program(code="def solve(x): return x + 1")
        insp2 = Program(code="def solve(x): return x * 2")

        result = SampleResult(
            parent=parent,
            inspirations=[insp1, insp2],
            metadata={"source_cell": 3},
        )

        assert result.parent == parent
        assert result.inspirations == [insp1, insp2]
        assert result.metadata == {"source_cell": 3}

    def test_sample_result_empty_inspirations(self):
        """SampleResult works with empty inspirations."""
        parent = Program(code="def solve(x): return x")

        result = SampleResult(parent=parent)

        assert result.parent == parent
        assert result.inspirations == []
        assert result.metadata == {}


class TestCVTMAPElitesPool:
    """Tests for CVTMAPElitesPool."""

    @pytest.fixture(autouse=True)
    def mock_centroid_init(self, monkeypatch):
        """Avoid sklearn KMeans in tests; keep deterministic centroid layout."""
        def _fake_init(self):
            rng = np.random.default_rng(42)
            return rng.uniform(0.0, 1.0, size=(self._n_centroids, self._n_dims))

        monkeypatch.setattr(CVTMAPElitesPool, "_init_cvt_centroids", _fake_init)

    @pytest.fixture
    def extractor(self):
        """Create a basic behavior extractor."""
        return BehaviorExtractor(
            ast_features=["code_length", "loop_count", "cyclomatic_complexity"]
        )

    @pytest.fixture
    def pool(self, extractor):
        """Create a pool with small config for testing."""
        return CVTMAPElitesPool(
            behavior_extractor=extractor,
            n_centroids=10,
            temperature=1.0,
        )

    def test_creation(self, extractor):
        """CVTMAPElitesPool can be created."""
        pool = CVTMAPElitesPool(
            behavior_extractor=extractor,
            n_centroids=100,
        )

        assert pool.size() == 0

    def test_add_valid_program(self, pool):
        """add() accepts valid programs."""
        prog = Program(code="def solve(x): return x")
        result = EvaluationResult(
            scores={"score": 0.8},
            is_valid=True,
        )

        accepted, _ = pool.add(prog, result)

        assert accepted is True
        assert pool.size() == 1

    def test_add_invalid_program(self, pool):
        """add() rejects invalid programs."""
        prog = Program(code="def solve(x): return x")
        result = EvaluationResult(
            is_valid=False,
            error="Error",
        )

        accepted, _ = pool.add(prog, result)

        assert accepted is False
        assert pool.size() == 0

    def test_add_replaces_worse_in_same_cell(self, pool):
        """add() replaces lower-scoring program in same cell."""
        # Two programs with same behavior should map to same cell
        prog1 = Program(code="def solve(x): return x")
        prog2 = Program(code="def solve(x): return x")  # Same code = same behavior

        pool.add(prog1, EvaluationResult(
                        scores={"score": 0.5},
            is_valid=True,
        ))
        pool.add(prog2, EvaluationResult(
                        scores={"score": 0.8},  # Higher score
            is_valid=True,
        ))

        # Should still be 1 (replaced, not added)
        assert pool.size() == 1

        # best should be the higher-scoring one
        best = pool.best()
        assert best.id == prog2.id

    def test_add_rejects_worse_in_same_cell(self, pool):
        """add() rejects lower-scoring program if cell has better."""
        prog1 = Program(code="def solve(x): return x")
        prog2 = Program(code="def solve(x): return x")

        pool.add(prog1, EvaluationResult(
            scores={"score": 0.8},  # Higher score first
            is_valid=True,
        ))

        accepted, _ = pool.add(prog2, EvaluationResult(
            scores={"score": 0.5},  # Lower score
            is_valid=True,
        ))

        assert accepted is False
        assert pool.size() == 1

    def test_sample_returns_sample_result(self, pool):
        """sample() returns a SampleResult."""
        prog = Program(code="def solve(x): return x")
        pool.add(prog, EvaluationResult(
            scores={"score": 0.8},
            is_valid=True,
        ))

        sample = pool.sample("ucb", n_parents=1)

        assert isinstance(sample, SampleResult)
        assert sample.parent is not None

    def test_sample_empty_raises(self, pool):
        """sample() raises when pool is empty."""
        with pytest.raises(ValueError, match="empty"):
            pool.sample("ucb")

    def test_best_returns_highest_score(self, pool):
        """best() returns best program in archive."""
        codes = [
            "def solve(x): return x",
            "def solve(x): return x + 1",
            '''def solve(x):
    for i in range(10):
        x += i
    return x''',
        ]

        for i, code in enumerate(codes):
            prog = Program(code=code)
            pool.add(prog, EvaluationResult(
                scores={"score": i * 0.3},  # Last has highest
                is_valid=True,
            ))

        best = pool.best()

        # The last program should have highest score
        assert "for i in range" in best.code

    def test_best_empty_raises(self, pool):
        """best() raises when pool is empty."""
        with pytest.raises(ValueError, match="empty"):
            pool.best()

    def test_on_generation_complete_increments_counter(self, pool):
        """on_generation_complete() increments generation counter."""
        initial_gen = pool.get_stats()["generation"]

        pool.on_generation_complete()

        assert pool.get_stats()["generation"] == initial_gen + 1

    def test_get_stats(self, pool):
        """get_stats() returns expected statistics."""
        prog = Program(code="def solve(x): return x")
        pool.add(prog, EvaluationResult(
            scores={"score": 0.8},
            is_valid=True,
        ))
        pool.on_generation_complete()

        stats = pool.get_stats()

        assert stats["archive_size"] == 1
        assert stats["n_centroids"] == 10
        assert stats["generation"] == 1
        assert "samplers" in stats

    def test_behavior_diversity(self, extractor):
        """Programs can be added to archive."""
        pool = CVTMAPElitesPool(
            behavior_extractor=extractor,
            n_centroids=1000,  # More centroids for finer granularity
        )

        # Add programs with different code structures
        codes = [
            # Simple: short, no loops, low complexity
            "def solve(x): return x",
            # Complex: longer, multiple loops, high complexity
            '''def solve(x):
    result = 0
    for i in range(100):
        for j in range(100):
            if i > j:
                result += i * j
            elif i < j:
                result -= i + j
            else:
                result *= 2
    return result''',
        ]

        for code in codes:
            prog = Program(code=code)
            pool.add(prog, EvaluationResult(
                scores={"score": 0.5},
                is_valid=True,
            ))

        # At least one should be accepted
        assert pool.size() >= 1

    def test_multiple_sampler_strategies(self, pool):
        """Pool has multiple sampling strategies."""
        # Add some programs
        for i in range(5):
            code = f"def solve(x): return x + {i}"
            prog = Program(code=code)
            pool.add(prog, EvaluationResult(
                scores={"score": i * 0.1},
                is_valid=True,
            ))

        # Check available samplers
        stats = pool.get_stats()
        assert "ucb" in stats["samplers"]
        assert "softmax" in stats["samplers"]
        assert "uniform" in stats["samplers"]


class TestPoolProtocolCompliance:
    """Tests that CVTMAPElitesPool implements the expected interface."""

    @pytest.fixture(autouse=True)
    def mock_centroid_init(self, monkeypatch):
        """Avoid sklearn KMeans in tests; keep deterministic centroid layout."""
        def _fake_init(self):
            rng = np.random.default_rng(42)
            return rng.uniform(0.0, 1.0, size=(self._n_centroids, self._n_dims))

        monkeypatch.setattr(CVTMAPElitesPool, "_init_cvt_centroids", _fake_init)

    @pytest.fixture
    def pool(self):
        """Create a pool instance."""
        extractor = BehaviorExtractor(
            ast_features=["code_length", "loop_count"]
        )
        return CVTMAPElitesPool(
            behavior_extractor=extractor,
            n_centroids=10,
        )

    def test_pool_has_add(self, pool):
        """Pool has add() method."""
        assert hasattr(pool, "add")
        assert callable(pool.add)

    def test_pool_has_sample(self, pool):
        """Pool has sample() method."""
        assert hasattr(pool, "sample")
        assert callable(pool.sample)

    def test_pool_has_best(self, pool):
        """Pool has best() method."""
        assert hasattr(pool, "best")
        assert callable(pool.best)

    def test_pool_has_size(self, pool):
        """Pool has size() method."""
        assert hasattr(pool, "size")
        assert callable(pool.size)

    def test_pool_has_lifecycle_methods(self, pool):
        """Pool has on_generation_complete()."""
        assert hasattr(pool, "on_generation_complete")
        assert callable(pool.on_generation_complete)
