"""
CVT-MAP-Elites Pool with Multi-Strategy Sampling.

Single shared archive with multiple sampling strategies:
- UCB (Upper Confidence Bound) - exploration/exploitation balance
- Softmax - temperature-based fitness-weighted sampling
- Uniform - random sampling for exploration
- Per-subscore - sample best performers on individual metrics
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sklearn.cluster import KMeans

from ..core import Program, EvaluationResult
from ..behavior import BehaviorExtractor, FeatureVector
from .protocol import SampleResult


@dataclass
class Elite:
    """An elite program occupying a cell."""
    program: Program
    result: EvaluationResult
    behavior: FeatureVector


@dataclass
class CellStats:
    """Statistics for a cell used by samplers."""
    n_samples: int = 0      # Times this cell was sampled
    n_successes: int = 0    # Times sampling led to accepted offspring
    total_reward: float = 0.0  # Cumulative reward for UCB

    def success_rate(self) -> float:
        if self.n_samples == 0:
            return 0.5  # Prior
        return self.n_successes / self.n_samples

    def ucb_score(self, total_samples: int, c: float = 2.0) -> float:
        """UCB1 score: exploitation + exploration bonus."""
        if self.n_samples == 0:
            return float('inf')  # Unexplored cells have infinite priority
        exploitation = self.total_reward / self.n_samples
        exploration = c * math.sqrt(math.log(total_samples + 1) / self.n_samples)
        return exploitation + exploration


class Sampler(ABC):
    """Abstract base class for sampling strategies."""

    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type  # "light" or "heavy"
        self.cell_stats: dict[int, CellStats] = {}

    def get_or_create_stats(self, cell: int) -> CellStats:
        if cell not in self.cell_stats:
            self.cell_stats[cell] = CellStats()
        return self.cell_stats[cell]

    def update(self, cell: int, success: bool, reward: float = 0.0) -> None:
        """Update statistics after observing outcome."""
        stats = self.get_or_create_stats(cell)
        stats.n_samples += 1
        if success:
            stats.n_successes += 1
        stats.total_reward += reward

    @abstractmethod
    def select_cells(self, elites: dict[int, Elite], n: int) -> list[int]:
        """Select n cells from the archive."""
        pass

    def get_stats_summary(self) -> dict:
        if not self.cell_stats:
            return {"n_cells": 0}
        rates = [s.success_rate() for s in self.cell_stats.values()]
        return {
            "n_cells": len(self.cell_stats),
            "total_samples": sum(s.n_samples for s in self.cell_stats.values()),
            "mean_success_rate": sum(rates) / len(rates) if rates else 0.0,
        }


class UCBSampler(Sampler):
    """Upper Confidence Bound sampling - balances exploration and exploitation."""

    def __init__(self, c: float = 2.0):
        super().__init__("ucb", "light")
        self.c = c
        self._total_samples = 0

    def select_cells(self, elites: dict[int, Elite], n: int) -> list[int]:
        if not elites:
            return []
        self._total_samples += 1
        cells = list(elites.keys())

        # Compute UCB scores
        scores = []
        for cell in cells:
            stats = self.get_or_create_stats(cell)
            scores.append((stats.ucb_score(self._total_samples, self.c), cell))

        # Sort by UCB score descending, take top n
        scores.sort(reverse=True, key=lambda x: x[0])
        return [cell for _, cell in scores[:min(n, len(cells))]]


class SoftmaxSampler(Sampler):
    """Temperature-based softmax sampling weighted by fitness."""

    def __init__(self, temperature: float = 1.0):
        super().__init__("softmax", "heavy")
        self.temperature = temperature

    def select_cells(self, elites: dict[int, Elite], n: int) -> list[int]:
        if not elites:
            return []
        cells = list(elites.keys())
        scores = [elites[c].result.primary_score for c in cells]

        # Softmax weights
        max_s = max(scores)
        exp_s = [math.exp((s - max_s) / self.temperature) for s in scores]
        total = sum(exp_s)
        weights = [e / total for e in exp_s]

        # Weighted sampling without replacement
        selected = []
        remaining_cells = cells.copy()
        remaining_weights = weights.copy()

        for _ in range(min(n, len(cells))):
            if not remaining_cells:
                break
            # Normalize remaining weights
            w_sum = sum(remaining_weights)
            if w_sum == 0:
                break
            probs = [w / w_sum for w in remaining_weights]
            idx = np.random.choice(len(remaining_cells), p=probs)
            selected.append(remaining_cells[idx])
            remaining_cells.pop(idx)
            remaining_weights.pop(idx)

        return selected


class UniformSampler(Sampler):
    """Uniform random sampling for pure exploration."""

    def __init__(self):
        super().__init__("uniform", "light")

    def select_cells(self, elites: dict[int, Elite], n: int) -> list[int]:
        if not elites:
            return []
        cells = list(elites.keys())
        return random.sample(cells, min(n, len(cells)))


class SubscoreSampler(Sampler):
    """Sample cells weighted by a specific subscore metric using softmax."""

    def __init__(self, subscore_key: str, display_name: str, temperature: float = 1.0):
        super().__init__(f"subscore_{subscore_key}", "light")
        self.subscore_key = subscore_key
        self.display_name = display_name
        self.temperature = temperature

    def select_cells(self, elites: dict[int, Elite], n: int) -> list[int]:
        if not elites:
            return []

        # Get cells and their subscore values
        cells = list(elites.keys())
        scores = [elites[c].program.metadata.get(self.subscore_key, 0.0) for c in cells]

        # Softmax weighting by subscore
        max_s = max(scores) if scores else 0
        exp_s = [math.exp((s - max_s) / self.temperature) for s in scores]
        total = sum(exp_s)
        weights = [e / total for e in exp_s] if total > 0 else [1.0 / len(cells)] * len(cells)

        # Weighted sampling without replacement
        selected = []
        remaining_cells = cells.copy()
        remaining_weights = weights.copy()

        for _ in range(min(n, len(cells))):
            if not remaining_cells:
                break
            w_sum = sum(remaining_weights)
            if w_sum == 0:
                break
            probs = [w / w_sum for w in remaining_weights]
            idx = np.random.choice(len(remaining_cells), p=probs)
            selected.append(remaining_cells[idx])
            remaining_cells.pop(idx)
            remaining_weights.pop(idx)

        return selected


class CVTMAPElitesPool:
    """
    CVT-MAP-Elites Pool with Multi-Strategy Sampling.

    Single shared archive with multiple sampling strategies.
    Each strategy can be associated with different LLM models.
    """

    def __init__(
        self,
        behavior_extractor: BehaviorExtractor,
        n_centroids: int = 1000,
        temperature: float = 1.0,
        bounds_padding: float = 0.1,
        subscore_keys: Optional[list[str]] = None,
        defer_centroids: bool = False,
    ) -> None:
        self._extractor = behavior_extractor
        self._n_centroids = n_centroids
        self._temperature = temperature
        self._feature_names = behavior_extractor.features
        self._n_dims = len(self._feature_names)
        self._bounds_padding = bounds_padding

        # Adaptive bounds
        self._mins: Optional[np.ndarray] = None
        self._maxs: Optional[np.ndarray] = None
        self._ranges: Optional[np.ndarray] = None

        # Initialize centroids (can be deferred for data-driven initialization)
        self._centroids: Optional[np.ndarray] = None
        if not defer_centroids:
            self._centroids = self._init_cvt_centroids()

        # Single shared archive
        self._elites: dict[int, Elite] = {}
        self._best_score: float = float('-inf')
        self._generation = 0

        # Initialize samplers
        self._samplers: dict[str, Sampler] = {
            "ucb": UCBSampler(c=2.0),
            "softmax": SoftmaxSampler(temperature=temperature),
            "uniform": UniformSampler(),
        }

        # Add per-subscore samplers
        if subscore_keys:
            for key in subscore_keys:
                sampler = SubscoreSampler(key, key)
                self._samplers[f"subscore_{key}"] = sampler

    def _init_cvt_centroids(self) -> np.ndarray:
        """Initialize CVT centroids using k-means++ in normalized space."""
        n_dims = len(self._feature_names)
        n_samples = max(10000, self._n_centroids * 10)
        samples = np.random.uniform(0, 1, size=(n_samples, n_dims))

        kmeans = KMeans(
            n_clusters=self._n_centroids,
            init='k-means++',
            n_init=1,
            max_iter=100,
            random_state=42
        )
        kmeans.fit(samples)
        return kmeans.cluster_centers_

    def set_centroids_from_data(
        self,
        behavior_vectors: list[np.ndarray],
        percentile_low: float = 5.0,
        percentile_high: float = 95.0,
        n_centroids: int = 50,
    ) -> int:
        """
        Set centroids from actual behavior data using k-means clustering.

        Args:
            behavior_vectors: List of raw behavior vectors (not normalized)
            percentile_low: Lower percentile to exclude (default 5%)
            percentile_high: Upper percentile to exclude (default 95%)
            n_centroids: Number of centroids to create via k-means

        Returns:
            Number of centroids created
        """
        if len(behavior_vectors) < 3:
            raise ValueError("Need at least 3 behavior vectors to build centroids")

        data = np.array(behavior_vectors)

        # Compute percentile bounds for each dimension
        low_bounds = np.percentile(data, percentile_low, axis=0)
        high_bounds = np.percentile(data, percentile_high, axis=0)

        # Filter out outliers
        mask = np.all((data >= low_bounds) & (data <= high_bounds), axis=1)
        filtered_data = data[mask]

        # Use original data if filtering removes too many points for desired centroids
        if len(filtered_data) < n_centroids:
            filtered_data = data

        # Set bounds from filtered data
        self._mins = filtered_data.min(axis=0)
        self._maxs = filtered_data.max(axis=0)
        padding = np.maximum(np.abs(self._maxs - self._mins) * self._bounds_padding, 1e-6)
        self._mins -= padding
        self._maxs += padding
        self._ranges = self._maxs - self._mins
        self._ranges[self._ranges == 0] = 1

        # Normalize filtered data
        normalized_data = (filtered_data - self._mins) / self._ranges

        # Use k-means to create well-spread centroids in the actual behavior space
        actual_n_centroids = min(n_centroids, len(filtered_data))
        kmeans = KMeans(
            n_clusters=actual_n_centroids,
            init='k-means++',
            n_init=3,
            max_iter=100,
            random_state=42
        )
        kmeans.fit(normalized_data)
        self._centroids = kmeans.cluster_centers_
        self._n_centroids = actual_n_centroids

        return self._n_centroids

    @staticmethod
    def select_most_diverse(
        behavior_vectors: list[np.ndarray],
        k: int,
    ) -> list[int]:
        """
        Select k most diverse indices using farthest-first traversal.

        This guarantees maximum spread in behavior space, not just highest scores.

        Args:
            behavior_vectors: List of behavior vectors
            k: Number of diverse items to select

        Returns:
            List of indices of the k most diverse items
        """
        n = len(behavior_vectors)
        if n <= k:
            return list(range(n))

        behaviors = np.array(behavior_vectors)

        # Normalize for fair distance computation
        mins = behaviors.min(axis=0)
        maxs = behaviors.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        normalized = (behaviors - mins) / ranges

        # Farthest-first traversal
        selected = [0]  # Start with first item
        min_distances = np.full(n, np.inf)

        for _ in range(k - 1):
            # Update min distances to selected set
            last_selected = normalized[selected[-1]]
            for i in range(n):
                dist = np.linalg.norm(normalized[i] - last_selected)
                min_distances[i] = min(min_distances[i], dist)

            # Exclude already selected
            min_distances[selected] = -np.inf

            # Pick farthest from selected set
            next_idx = int(np.argmax(min_distances))
            selected.append(next_idx)

        return selected

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalize a feature vector to [0, 1] range."""
        if self._mins is None:
            return np.full(self._n_dims, 0.5)
        normalized = (vec - self._mins) / self._ranges
        return np.clip(normalized, 0, 1)

    def _behavior_to_normalized_vector(self, behavior: FeatureVector) -> np.ndarray:
        """Convert FeatureVector to normalized numpy array."""
        raw = np.array([behavior[f] for f in self._feature_names])
        return self._normalize(raw)

    def _find_nearest_centroid(self, behavior: FeatureVector) -> int:
        """Find nearest centroid in normalized space."""
        vec = self._behavior_to_normalized_vector(behavior)
        distances = np.sum((self._centroids - vec) ** 2, axis=1)
        return int(np.argmin(distances))

    def add(self, program: Program, evaluation_result: EvaluationResult) -> bool:
        """Add program to archive. Returns True if accepted."""
        if not evaluation_result.is_valid:
            return False

        behavior = self._extractor.extract(program)
        cell_index = self._find_nearest_centroid(behavior)
        new_score = evaluation_result.primary_score

        if cell_index not in self._elites:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior)
            self._best_score = max(self._best_score, new_score)
            return True

        if new_score > self._elites[cell_index].result.primary_score:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior)
            self._best_score = max(self._best_score, new_score)
            return True

        return False

    def sample(
        self,
        sampler_name: str,
        n_parents: int = 4,
        context: Optional[dict] = None,
    ) -> SampleResult:
        """Sample from archive using specified strategy."""
        if sampler_name not in self._samplers:
            raise ValueError(f"Unknown sampler: {sampler_name}")

        sampler = self._samplers[sampler_name]

        if not self._elites:
            raise ValueError("Archive is empty")

        cells = sampler.select_cells(self._elites, n_parents)

        if not cells:
            # Fallback to uniform if sampler returns nothing
            cells = random.sample(list(self._elites.keys()), min(n_parents, len(self._elites)))

        return SampleResult(
            parent=self._elites[cells[0]].program,
            inspirations=[self._elites[c].program for c in cells[1:]],
            metadata={
                "sampler": sampler_name,
                "source_cell": cells[0],
                "model_type": sampler.model_type,
            },
        )

    def update_sampler(self, sampler_name: str, cell: int, success: bool, reward: float = 0.0) -> None:
        """Update sampler statistics after observing outcome."""
        if sampler_name in self._samplers:
            self._samplers[sampler_name].update(cell, success, reward)

    def get_sampler_names(self) -> list[str]:
        """Get list of all sampler names."""
        return list(self._samplers.keys())

    def get_sampler(self, name: str) -> Sampler:
        """Get sampler by name."""
        return self._samplers[name]

    def best(self, metric: str = "score") -> Program:
        """Return best program in archive."""
        if not self._elites:
            raise ValueError("Archive is empty")

        best_elite = max(self._elites.values(), key=lambda e: e.result.primary_score)
        return best_elite.program

    def size(self) -> int:
        return len(self._elites)

    def on_generation_complete(self) -> None:
        self._generation += 1

    def get_stats(self) -> dict:
        stats = {
            "archive_size": self.size(),
            "n_centroids": self._n_centroids,
            "best_score": self._best_score,
            "generation": self._generation,
            "samplers": {
                name: sampler.get_stats_summary()
                for name, sampler in self._samplers.items()
            },
        }
        if self._mins is not None:
            stats["learned_bounds"] = {
                f: (float(self._mins[i]), float(self._maxs[i]))
                for i, f in enumerate(self._feature_names)
            }
        return stats
