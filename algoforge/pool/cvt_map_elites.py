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
    raw_behavior: Optional[dict] = None  # Raw feature values for cross-island migration


@dataclass
class CellStats:
    """Statistics for a cell used by samplers."""
    n_samples: int = 0      # Times this cell was sampled
    n_successes: int = 0    # Times sampling led to accepted offspring

    def success_rate(self) -> float:
        if self.n_samples == 0:
            return 0.5  # Prior
        return self.n_successes / self.n_samples

    def ucb_score(self, total_samples: int, c: float = 2.0) -> float:
        """UCB1 score: exploitation (success rate) + exploration bonus.

        Uses acceptance rate rather than raw scores to avoid bias toward
        high-scoring cells that produce rejected clones. This encourages
        sampling cells that actually improve the archive.
        """
        if self.n_samples == 0:
            return float('inf')  # Unexplored cells have infinite priority
        exploitation = self.success_rate()
        exploration = c * math.sqrt(math.log(total_samples + 1) / self.n_samples)
        return exploitation + exploration


class Sampler(ABC):
    """Abstract base class for sampling strategies."""

    def __init__(self, name: str):
        self.name = name
        self.cell_stats: dict[int, CellStats] = {}

    def get_or_create_stats(self, cell: int) -> CellStats:
        if cell not in self.cell_stats:
            self.cell_stats[cell] = CellStats()
        return self.cell_stats[cell]

    def update(self, cell: int, success: bool) -> None:
        """Update statistics after observing outcome."""
        stats = self.get_or_create_stats(cell)
        stats.n_samples += 1
        if success:
            stats.n_successes += 1

    @abstractmethod
    def select_cells(
        self, elites: dict[int, Elite], n: int, context: Optional[dict] = None
    ) -> list[int]:
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
        super().__init__("ucb")
        self.c = c
        self._total_samples = 0

    def select_cells(
        self, elites: dict[int, Elite], n: int, context: Optional[dict] = None
    ) -> list[int]:
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
        super().__init__("softmax")
        self.temperature = temperature

    def select_cells(
        self, elites: dict[int, Elite], n: int, context: Optional[dict] = None
    ) -> list[int]:
        if not elites:
            return []
        cells = list(elites.keys())
        scores = [elites[c].result.primary_score for c in cells]

        min_s = min(scores)
        max_s = max(scores)
        score_range = max_s - min_s if max_s > min_s else 1.0
        normalized = [(s - min_s) / score_range for s in scores]  # [0, 1]

        exp_s = [math.exp((ns - 1.0) / self.temperature) for ns in normalized]
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


class CyclicAnnealingSampler(Sampler):
    """
    Cyclic annealing sampler with budget-based temperature schedule.

    Temperature cycles from T_max to T_min multiple times during the run:
        T = T_min + (T_max - T_min) * (1 - cycle_progress)
    where cycle_progress = (budget_progress * n_cycles) mod 1

    This provides periodic exploration/exploitation phases:
    - High temperature: more uniform sampling (exploration)
    - Low temperature: favor high-scoring cells (exploitation)
    """

    def __init__(self, t_max: float = 1.2, t_min: float = 0.15, n_cycles: int = 4):
        super().__init__("cyclic_annealing")
        self.t_max = t_max
        self.t_min = t_min
        self.n_cycles = n_cycles
        self._last_temperature: float = t_max

    def _compute_temperature(self, budget_progress: float) -> float:
        """Compute temperature based on budget progress (0 to 1)."""
        cycle_progress = (budget_progress * self.n_cycles) % 1.0
        temperature = self.t_min + (self.t_max - self.t_min) * (1.0 - cycle_progress)
        self._last_temperature = temperature
        return temperature

    def select_cells(
        self, elites: dict[int, Elite], n: int, context: Optional[dict] = None
    ) -> list[int]:
        if not elites:
            return []

        # Get budget progress from context, default to 0 (start of run)
        budget_progress = 0.0
        if context and "budget_progress" in context:
            budget_progress = context["budget_progress"]

        temperature = self._compute_temperature(budget_progress)

        cells = list(elites.keys())
        scores = [elites[c].result.primary_score for c in cells]

        min_s = min(scores)
        max_s = max(scores)
        score_range = max_s - min_s if max_s > min_s else 1.0
        normalized = [(s - min_s) / score_range for s in scores]  # [0, 1]

        exp_s = [math.exp((ns - 1.0) / temperature) for ns in normalized]
        total = sum(exp_s)
        weights = [e / total for e in exp_s]

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

    def get_stats_summary(self) -> dict:
        stats = super().get_stats_summary()
        stats["last_temperature"] = self._last_temperature
        stats["t_max"] = self.t_max
        stats["t_min"] = self.t_min
        stats["n_cycles"] = self.n_cycles
        return stats


class UniformSampler(Sampler):
    """Uniform random sampling for pure exploration."""

    def __init__(self):
        super().__init__("uniform")

    def select_cells(self, elites: dict[int, Elite], n: int, context: Optional[dict] = None) -> list[int]:
        if not elites:
            return []
        cells = list(elites.keys())
        return random.sample(cells, min(n, len(cells)))


class SubscoreSampler(Sampler):
    """Sample cells weighted by a specific subscore metric using softmax."""

    def __init__(self, subscore_key: str, display_name: str, temperature: float = 1.0):
        super().__init__(f"subscore_{subscore_key}")
        self.subscore_key = subscore_key
        self.display_name = display_name
        self.temperature = temperature

    def select_cells(
        self, elites: dict[int, Elite], n: int, context: Optional[dict] = None
    ) -> list[int]:
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


@dataclass
class SamplerModelConfig:
    sampler_name: str
    model: str
    weight: float = 1.0


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
        self._feature_names = behavior_extractor.features
        self._n_dims = len(self._feature_names)

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
            "cyclic_annealing": CyclicAnnealingSampler(),
        }

        # Add per-subscore samplers
        if subscore_keys:
            for key in subscore_keys:
                sampler = SubscoreSampler(key, key)
                self._samplers[f"subscore_{key}"] = sampler

        # Sampler-model pairs for weighted selection
        self._sampler_model_pairs: list[SamplerModelConfig] = []
        self._total_weight: float = 0.0

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
        n_centroids: int = 50,
    ) -> tuple[int, np.ndarray]:
        """
        Set centroids from behavior data using k-means clustering.

        Args:
            behavior_vectors: List of behavior vectors (already normalized via z-score+sigmoid)
            n_centroids: Number of centroids to create via k-means

        Returns:
            Tuple of (number of centroids created, labels array for each input vector)
        """
        if not behavior_vectors:
            raise ValueError("Need at least 1 behavior vector to build centroids")

        data = np.array(behavior_vectors)
        actual_n_centroids = min(n_centroids, len(data))

        kmeans = KMeans(
            n_clusters=actual_n_centroids,
            init='k-means++',
            n_init=3,
            max_iter=100,
            random_state=42
        )
        kmeans.fit(data)
        self._centroids = kmeans.cluster_centers_
        self._n_centroids = actual_n_centroids

        # No extra normalization - data is already [0,1] from z-score+sigmoid
        self._mins = np.zeros(self._n_dims)
        self._maxs = np.ones(self._n_dims)
        self._ranges = np.ones(self._n_dims)

        return self._n_centroids, kmeans.labels_

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

    def add(self, program: Program, evaluation_result: EvaluationResult) -> tuple[bool, int]:
        """Add program to archive. Returns (accepted, cell_index)."""
        if not evaluation_result.is_valid:
            return False, -1

        behavior = self._extractor.extract(program)
        raw_behavior = behavior.values.copy()  # Store for cross-island migration
        cell_index = self._find_nearest_centroid(behavior)
        new_score = evaluation_result.primary_score

        if cell_index not in self._elites:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior)
            self._best_score = max(self._best_score, new_score)
            return True, cell_index

        if new_score > self._elites[cell_index].result.primary_score:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior)
            self._best_score = max(self._best_score, new_score)
            return True, cell_index

        return False, cell_index

    def add_with_raw_behavior(
        self,
        program: Program,
        evaluation_result: EvaluationResult,
        raw_behavior: dict[str, float],
    ) -> bool:
        """
        Add a migrant program using raw behavior values for re-normalization.

        Used for cross-island migration where the source island has different
        adaptive bounds. The raw_behavior contains pre-normalized feature values
        that will be re-normalized using this island's bounds.
        """
        if not evaluation_result.is_valid:
            return False

        # Create FeatureVector from raw behavior
        behavior = FeatureVector(raw_behavior.copy())
        cell_index = self._find_nearest_centroid(behavior)
        new_score = evaluation_result.primary_score

        if cell_index not in self._elites:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior.copy())
            self._best_score = max(self._best_score, new_score)
            return True

        if new_score > self._elites[cell_index].result.primary_score:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior.copy())
            self._best_score = max(self._best_score, new_score)
            return True

        return False

    def add_at_cell(
        self,
        cell_index: int,
        program: Program,
        evaluation_result: EvaluationResult,
        behavior: FeatureVector,
    ) -> bool:
        """
        Add elite directly at a specific cell index (no re-extraction).

        Used during init when we already have behavior vectors and k-means labels.
        Only accepts if cell is empty or new score beats existing.
        """
        if not evaluation_result.is_valid:
            return False

        new_score = evaluation_result.primary_score
        raw_behavior = behavior.values.copy()

        if cell_index not in self._elites:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior)
            self._best_score = max(self._best_score, new_score)
            return True

        if new_score > self._elites[cell_index].result.primary_score:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior)
            self._best_score = max(self._best_score, new_score)
            return True

        return False

    def add_with_behavior_noise(
        self,
        program: Program,
        evaluation_result: EvaluationResult,
        noise_scale: float = 0.05,
    ) -> tuple[bool, int]:
        """
        Add program with noise applied to behavior vector before cell assignment.

        This allows PE solutions to potentially land in nearby cells,
        increasing exploration of the behavior space.

        Args:
            program: The program to add
            evaluation_result: Evaluation results for the program
            noise_scale: Standard deviation of Gaussian noise to add (default 0.05)

        Returns:
            Tuple of (accepted: bool, cell_index: int)
        """
        if not evaluation_result.is_valid:
            return False, -1

        behavior = self._extractor.extract(program)

        # Apply noise to normalized behavior before cell assignment
        noisy_values = {}
        for feature in self._feature_names:
            original = behavior[feature]
            noise = np.random.normal(0, noise_scale)
            noisy_values[feature] = float(np.clip(original + noise, 0.0, 1.0))

        noisy_behavior = FeatureVector(noisy_values)
        cell_index = self._find_nearest_centroid(noisy_behavior)

        # Store original behavior (not noisy) for the elite record
        raw_behavior = behavior.values.copy()
        new_score = evaluation_result.primary_score

        if cell_index not in self._elites:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior)
            self._best_score = max(self._best_score, new_score)
            return True, cell_index

        if new_score > self._elites[cell_index].result.primary_score:
            self._elites[cell_index] = Elite(program, evaluation_result, behavior, raw_behavior)
            self._best_score = max(self._best_score, new_score)
            return True, cell_index

        return False, cell_index

    def get_elites(self) -> dict[int, Elite]:
        """Get all elites in the archive."""
        return self._elites

    def get_elite(self, cell_index: int) -> Optional[Elite]:
        """Get elite at a specific cell index."""
        return self._elites.get(cell_index)

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

        cells = sampler.select_cells(self._elites, n_parents, context)

        if not cells:
            # Fallback to uniform if sampler returns nothing
            cells = random.sample(list(self._elites.keys()), min(n_parents, len(self._elites)))

        return SampleResult(
            parent=self._elites[cells[0]].program,
            inspirations=[self._elites[c].program for c in cells[1:]],
            metadata={
                "sampler": sampler_name,
                "source_cell": cells[0],
            },
        )

    def update_sampler(self, sampler_name: str, cell: int, success: bool) -> None:
        """Update sampler statistics after observing outcome."""
        if sampler_name in self._samplers:
            self._samplers[sampler_name].update(cell, success)

    def get_sampler_names(self) -> list[str]:
        """Get list of all sampler names."""
        return list(self._samplers.keys())

    def get_sampler(self, name: str) -> Sampler:
        """Get sampler by name."""
        return self._samplers[name]

    def register_sampler_model_pair(
        self, sampler_name: str, model: str, weight: float = 1.0,
        temperature: Optional[float] = None, n_cycles: Optional[int] = None
    ) -> None:
        if weight <= 0:
            raise ValueError("Weight must be positive")

        actual_sampler_name = sampler_name

        # For softmax with custom temperature, create a new sampler instance
        if sampler_name == "softmax" and temperature is not None:
            actual_sampler_name = f"softmax_T{temperature}"
            if actual_sampler_name not in self._samplers:
                self._samplers[actual_sampler_name] = SoftmaxSampler(temperature=temperature)
        # For cyclic_annealing with custom n_cycles, create a new sampler instance
        elif sampler_name == "cyclic_annealing" and n_cycles is not None:
            actual_sampler_name = f"cyclic_annealing_C{n_cycles}"
            if actual_sampler_name not in self._samplers:
                self._samplers[actual_sampler_name] = CyclicAnnealingSampler(n_cycles=n_cycles)
        elif sampler_name not in self._samplers:
            raise ValueError(f"Unknown sampler: {sampler_name}. Available: {list(self._samplers.keys())}")

        self._sampler_model_pairs.append(SamplerModelConfig(actual_sampler_name, model, weight))
        self._total_weight += weight

    def get_weighted_sampler_config(self) -> tuple[str, str]:
        if not self._sampler_model_pairs:
            raise ValueError("No sampler-model pairs registered. Call register_sampler_model_pair() first.")

        r = random.random() * self._total_weight
        cumulative = 0.0
        for pair in self._sampler_model_pairs:
            cumulative += pair.weight
            if r <= cumulative:
                return pair.sampler_name, pair.model

        last = self._sampler_model_pairs[-1]
        return last.sampler_name, last.model

    def best(self, metric: str = "score") -> Program:
        """Return best program in archive."""
        if not self._elites:
            raise ValueError("Archive is empty")

        best_elite = max(self._elites.values(), key=lambda e: e.result.primary_score)
        return best_elite.program

    def size(self) -> int:
        return len(self._elites)

    def clear(self) -> int:
        """
        Clear all elites from the archive.

        Returns the number of elites removed.
        """
        n_removed = len(self._elites)
        self._elites.clear()
        self._best_score = float('-inf')
        return n_removed

    def get_top_elites(self, n: int) -> list[Elite]:
        """
        Get top n elites by score.

        Returns list of Elite objects sorted by score descending.
        """
        if not self._elites:
            return []

        sorted_elites = sorted(
            self._elites.values(),
            key=lambda e: e.result.primary_score,
            reverse=True
        )
        return sorted_elites[:n]

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

    def get_archive_snapshot(self) -> dict:
        """
        Get a JSON-serializable snapshot of the entire archive state.

        Returns a dict containing:
        - metadata: archive stats and configuration
        - elites: list of all elite programs with their scores and behaviors
        - sampler_stats: per-sampler statistics
        """
        elites_data = []
        for cell_idx, elite in self._elites.items():
            elite_data = {
                "cell_index": cell_idx,
                "program_id": str(elite.program.id),
                "code": elite.program.code,
                "scores": elite.result.scores,
                "primary_score": elite.result.primary_score,
                "behavior": elite.behavior.values,
                "metadata": elite.program.metadata,
                "created_at": elite.program.created_at.isoformat() if elite.program.created_at else None,
            }
            elites_data.append(elite_data)

        # Sort by primary score descending
        elites_data.sort(key=lambda x: x["primary_score"], reverse=True)

        sampler_stats = {}
        for name, sampler in self._samplers.items():
            stats = sampler.get_stats_summary()
            # Add per-cell stats
            cell_stats = {}
            for cell, cs in sampler.cell_stats.items():
                cell_stats[str(cell)] = {
                    "n_samples": cs.n_samples,
                    "n_successes": cs.n_successes,
                    "success_rate": cs.success_rate(),
                }
            stats["cell_stats"] = cell_stats
            sampler_stats[name] = stats

        snapshot = {
            "metadata": {
                "archive_size": self.size(),
                "n_centroids": self._n_centroids,
                "best_score": self._best_score,
                "generation": self._generation,
                "feature_names": self._feature_names,
            },
            "elites": elites_data,
            "sampler_stats": sampler_stats,
        }

        if self._mins is not None:
            snapshot["metadata"]["learned_bounds"] = {
                f: {"min": float(self._mins[i]), "max": float(self._maxs[i])}
                for i, f in enumerate(self._feature_names)
            }

        return snapshot
