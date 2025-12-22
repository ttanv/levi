"""
Tests for pool module: SimplePool and CVTMAPElitesPool.

These are the "method-differentiating primitives" - different optimization
methods are distinguished primarily by their pool implementation.
"""

import pytest
import random

from algoforge.core import Program, EvaluationResult
from algoforge.pool import SimplePool, CVTMAPElitesPool, SampleResult
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
            metadata={"island_index": 3},
        )

        assert result.parent == parent
        assert result.inspirations == [insp1, insp2]
        assert result.metadata == {"island_index": 3}

    def test_sample_result_empty_inspirations(self):
        """SampleResult works with empty inspirations."""
        parent = Program(code="def solve(x): return x")

        result = SampleResult(parent=parent)

        assert result.parent == parent
        assert result.inspirations == []
        assert result.metadata == {}


class TestSimplePool:
    """Tests for SimplePool."""

    def test_creation_default(self):
        """SimplePool can be created with defaults."""
        pool = SimplePool()

        assert pool.size() == 0

    def test_creation_with_options(self):
        """SimplePool can be created with custom options."""
        pool = SimplePool(max_size=50, temperature=2.0)

        assert pool.size() == 0

    def test_add_valid_program(self):
        """add() accepts valid programs."""
        pool = SimplePool()
        prog = Program(code="def solve(x): return x")
        result = EvaluationResult(
            program_id=prog.id,
            scores={"score": 0.8},
            is_valid=True,
        )

        accepted = pool.add(prog, result)

        assert accepted is True
        assert pool.size() == 1

    def test_add_invalid_program(self):
        """add() rejects invalid programs."""
        pool = SimplePool()
        prog = Program(code="def solve(x): return x")
        result = EvaluationResult(
            program_id=prog.id,
            is_valid=False,
            error="Syntax error",
        )

        accepted = pool.add(prog, result)

        assert accepted is False
        assert pool.size() == 0

    def test_add_multiple_programs(self):
        """Multiple programs can be added."""
        pool = SimplePool()

        for i in range(5):
            prog = Program(code=f"def solve(x): return x + {i}")
            result = EvaluationResult(
                program_id=prog.id,
                scores={"score": i * 0.1},
                is_valid=True,
            )
            pool.add(prog, result)

        assert pool.size() == 5

    def test_add_prunes_when_over_max_size(self):
        """add() prunes worst program when over max_size."""
        pool = SimplePool(max_size=3)

        programs = []
        for i in range(5):
            prog = Program(code=f"def solve(x): return x + {i}")
            result = EvaluationResult(
                program_id=prog.id,
                scores={"score": i * 0.2},  # Higher score for later programs
                is_valid=True,
            )
            pool.add(prog, result)
            programs.append(prog)

        # Should have pruned to max_size
        assert pool.size() == 3

    def test_sample_returns_sample_result(self):
        """sample() returns a SampleResult."""
        pool = SimplePool()
        prog = Program(code="def solve(x): return x")
        result = EvaluationResult(
            program_id=prog.id,
            scores={"score": 0.8},
            is_valid=True,
        )
        pool.add(prog, result)

        sample = pool.sample()

        assert isinstance(sample, SampleResult)
        assert sample.parent.code == prog.code

    def test_sample_empty_pool_raises(self):
        """sample() raises when pool is empty."""
        pool = SimplePool()

        with pytest.raises(ValueError, match="empty"):
            pool.sample()

    def test_sample_n_parents(self):
        """sample() returns requested number of parents."""
        pool = SimplePool()

        for i in range(5):
            prog = Program(code=f"def solve(x): return x + {i}")
            result = EvaluationResult(
                program_id=prog.id,
                scores={"score": i * 0.1},
                is_valid=True,
            )
            pool.add(prog, result)

        sample = pool.sample(n_parents=3)

        # 1 parent + 2 inspirations = 3 total
        assert len(sample.inspirations) == 2

    def test_sample_fitness_weighted(self):
        """sample() is biased toward higher-scoring programs."""
        pool = SimplePool(temperature=0.1)  # Low temp = more deterministic

        # Add low-scoring program
        prog_low = Program(code="def solve(x): return x")
        pool.add(prog_low, EvaluationResult(
            program_id=prog_low.id,
            scores={"score": 0.1},
            is_valid=True,
        ))

        # Add high-scoring program
        prog_high = Program(code="def solve(x): return x * 2")
        pool.add(prog_high, EvaluationResult(
            program_id=prog_high.id,
            scores={"score": 10.0},
            is_valid=True,
        ))

        # Sample many times
        high_count = 0
        for _ in range(100):
            sample = pool.sample(n_parents=1)
            if sample.parent.code == prog_high.code:
                high_count += 1

        # Should strongly prefer high-scoring program
        assert high_count > 80

    def test_best_returns_highest_score(self):
        """best() returns the highest-scoring program."""
        pool = SimplePool()

        for i in range(5):
            prog = Program(code=f"def solve(x): return x + {i}")
            result = EvaluationResult(
                program_id=prog.id,
                scores={"score": i * 0.2},
                is_valid=True,
            )
            pool.add(prog, result)

        best = pool.best()

        assert best.code == "def solve(x): return x + 4"

    def test_best_empty_pool_raises(self):
        """best() raises when pool is empty."""
        pool = SimplePool()

        with pytest.raises(ValueError, match="empty"):
            pool.best()

    def test_on_generation_complete_noop(self):
        """on_generation_complete() is a no-op for SimplePool."""
        pool = SimplePool()

        # Should not raise
        pool.on_generation_complete()

    def test_on_epoch_noop(self):
        """on_epoch() is a no-op for SimplePool."""
        pool = SimplePool()

        # Should not raise
        pool.on_epoch()


class TestCVTMAPElitesPool:
    """Tests for CVTMAPElitesPool."""

    @pytest.fixture
    def extractor(self):
        """Create a basic behavior extractor."""
        return BehaviorExtractor(
            features=["code_length", "loop_count", "cyclomatic_complexity"]
        )

    @pytest.fixture
    def pool(self, extractor):
        """Create a pool with small config for testing."""
        return CVTMAPElitesPool(
            behavior_extractor=extractor,
            n_centroids=10,
            n_islands=3,
            temperature=1.0,
        )

    def test_creation(self, extractor):
        """CVTMAPElitesPool can be created."""
        pool = CVTMAPElitesPool(
            behavior_extractor=extractor,
            n_centroids=100,
            n_islands=5,
        )

        assert pool.size() == 0

    def test_add_valid_program(self, pool):
        """add() accepts valid programs."""
        prog = Program(
            code="def solve(x): return x",
            metadata={"island_index": 0},
        )
        result = EvaluationResult(
            program_id=prog.id,
            scores={"score": 0.8},
            is_valid=True,
        )

        accepted = pool.add(prog, result)

        assert accepted is True
        assert pool.size() == 1

    def test_add_invalid_program(self, pool):
        """add() rejects invalid programs."""
        prog = Program(code="def solve(x): return x")
        result = EvaluationResult(
            program_id=prog.id,
            is_valid=False,
            error="Error",
        )

        accepted = pool.add(prog, result)

        assert accepted is False
        assert pool.size() == 0

    def test_add_replaces_worse_in_same_cell(self, pool):
        """add() replaces lower-scoring program in same cell."""
        # Two programs with same behavior should map to same cell
        prog1 = Program(
            code="def solve(x): return x",
            metadata={"island_index": 0},
        )
        prog2 = Program(
            code="def solve(x): return x",  # Same code = same behavior
            metadata={"island_index": 0},
        )

        pool.add(prog1, EvaluationResult(
            program_id=prog1.id,
            scores={"score": 0.5},
            is_valid=True,
        ))
        pool.add(prog2, EvaluationResult(
            program_id=prog2.id,
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
        prog1 = Program(
            code="def solve(x): return x",
            metadata={"island_index": 0},
        )
        prog2 = Program(
            code="def solve(x): return x",
            metadata={"island_index": 0},
        )

        pool.add(prog1, EvaluationResult(
            program_id=prog1.id,
            scores={"score": 0.8},  # Higher score first
            is_valid=True,
        ))

        accepted = pool.add(prog2, EvaluationResult(
            program_id=prog2.id,
            scores={"score": 0.5},  # Lower score
            is_valid=True,
        ))

        assert accepted is False
        assert pool.size() == 1

    def test_add_to_different_islands(self, pool):
        """Programs with different island_index go to different islands."""
        for i in range(3):
            prog = Program(
                code=f"def solve(x): return x + {i}",
                metadata={"island_index": i},
            )
            pool.add(prog, EvaluationResult(
                program_id=prog.id,
                scores={"score": 0.5},
                is_valid=True,
            ))

        stats = pool.get_stats()
        # Each island should have 1 elite
        assert all(size == 1 for size in stats["island_sizes"])

    def test_sample_returns_sample_result(self, pool):
        """sample() returns a SampleResult with island metadata."""
        prog = Program(
            code="def solve(x): return x",
            metadata={"island_index": 0},
        )
        pool.add(prog, EvaluationResult(
            program_id=prog.id,
            scores={"score": 0.8},
            is_valid=True,
        ))

        sample = pool.sample(n_parents=1)

        assert isinstance(sample, SampleResult)
        assert "island_index" in sample.metadata

    def test_sample_empty_raises(self, pool):
        """sample() raises when pool is empty."""
        with pytest.raises(ValueError, match="empty"):
            pool.sample()

    def test_sample_finds_non_empty_island(self, pool):
        """sample() finds a non-empty island if selected island is empty."""
        # Only add to island 0
        prog = Program(
            code="def solve(x): return x",
            metadata={"island_index": 0},
        )
        pool.add(prog, EvaluationResult(
            program_id=prog.id,
            scores={"score": 0.8},
            is_valid=True,
        ))

        # Should still be able to sample even if random selects empty island
        random.seed(42)  # May select empty island
        for _ in range(10):
            sample = pool.sample()
            assert sample.parent is not None

    def test_best_returns_global_best(self, pool):
        """best() returns best program across all islands."""
        for i in range(3):
            prog = Program(
                code=f"def solve(x): return x + {i}",
                metadata={"island_index": i},
            )
            pool.add(prog, EvaluationResult(
                program_id=prog.id,
                scores={"score": i * 0.3},  # Island 2 has highest
                is_valid=True,
            ))

        best = pool.best()

        assert best.code == "def solve(x): return x + 2"

    def test_best_empty_raises(self, pool):
        """best() raises when pool is empty."""
        with pytest.raises(ValueError, match="empty"):
            pool.best()

    def test_on_generation_complete_increments_counter(self, pool):
        """on_generation_complete() increments generation counter."""
        initial_gen = pool.get_stats()["generation"]

        pool.on_generation_complete()

        assert pool.get_stats()["generation"] == initial_gen + 1

    def test_on_epoch_culls_islands(self, pool):
        """on_epoch() culls bottom half of islands and reseeds."""
        # Add programs to each island with different scores
        for i in range(3):
            prog = Program(
                code=f"def solve(x): return x + {i}",
                metadata={"island_index": i},
            )
            pool.add(prog, EvaluationResult(
                program_id=prog.id,
                scores={"score": i * 0.3},  # Island 2 is best
                is_valid=True,
            ))

        pool.on_epoch()

        # After epoch, culled islands should be reseeded from survivors
        # Pool should still have programs
        assert pool.size() >= 1

    def test_get_stats(self, pool):
        """get_stats() returns expected statistics."""
        prog = Program(
            code="def solve(x): return x",
            metadata={"island_index": 0},
        )
        pool.add(prog, EvaluationResult(
            program_id=prog.id,
            scores={"score": 0.8},
            is_valid=True,
        ))
        pool.on_generation_complete()

        stats = pool.get_stats()

        assert stats["total_elites"] == 1
        assert stats["n_islands"] == 3
        assert stats["n_centroids"] == 10
        assert len(stats["island_sizes"]) == 3
        assert len(stats["island_best_scores"]) == 3
        assert stats["generation"] == 1

    def test_behavior_diversity(self, extractor):
        """Programs with very different behaviors likely occupy different cells."""
        pool = CVTMAPElitesPool(
            behavior_extractor=extractor,
            n_centroids=1000,  # More centroids for finer granularity
            n_islands=1,
        )

        # Add programs with VERY different code structures
        # These should have distinctly different behavioral feature vectors
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
            prog = Program(code=code, metadata={"island_index": 0})
            pool.add(prog, EvaluationResult(
                program_id=prog.id,
                scores={"score": 0.5},
                is_valid=True,
            ))

        # Both should be accepted (very different behavioral profiles)
        assert pool.size() == 2

    def test_random_island_assignment_when_no_metadata(self, pool):
        """Programs without island_index get random island assignment."""
        prog = Program(code="def solve(x): return x")  # No island_index

        accepted = pool.add(prog, EvaluationResult(
            program_id=prog.id,
            scores={"score": 0.8},
            is_valid=True,
        ))

        assert accepted is True
        assert pool.size() == 1


class TestPoolProtocolCompliance:
    """Tests that pools implement the protocol correctly."""

    @pytest.fixture
    def pools(self):
        """Create instances of both pool types."""
        extractor = BehaviorExtractor(
            features=["code_length", "loop_count"]
        )
        return [
            SimplePool(),
            CVTMAPElitesPool(
                behavior_extractor=extractor,
                n_centroids=10,
                n_islands=2,
            ),
        ]

    def test_all_pools_have_add(self, pools):
        """All pools have add() method."""
        for pool in pools:
            assert hasattr(pool, "add")
            assert callable(pool.add)

    def test_all_pools_have_sample(self, pools):
        """All pools have sample() method."""
        for pool in pools:
            assert hasattr(pool, "sample")
            assert callable(pool.sample)

    def test_all_pools_have_best(self, pools):
        """All pools have best() method."""
        for pool in pools:
            assert hasattr(pool, "best")
            assert callable(pool.best)

    def test_all_pools_have_size(self, pools):
        """All pools have size() method."""
        for pool in pools:
            assert hasattr(pool, "size")
            assert callable(pool.size)

    def test_all_pools_have_lifecycle_methods(self, pools):
        """All pools have on_generation_complete() and on_epoch()."""
        for pool in pools:
            assert hasattr(pool, "on_generation_complete")
            assert hasattr(pool, "on_epoch")
