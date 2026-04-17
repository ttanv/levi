"""BehaviorExtractor: Computes behavioral features from programs."""

import ast
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import Program
from .features import (
    compute_ast_depth,
    compute_branch_count,
    compute_call_count,
    compute_code_length,
    compute_comparison_count,
    compute_comprehension_count,
    compute_cyclomatic_complexity,
    compute_function_def_count,
    compute_loop_count,
    compute_loop_nesting_max,
    compute_math_operators,
    compute_numeric_literal_count,
    compute_range_max_arg,
    compute_subscript_count,
)


@dataclass
class FeatureVector:
    values: dict[str, float]

    def to_array(self, feature_names: list[str]) -> list[float]:
        return [self.values.get(name, 0.0) for name in feature_names]

    def __getitem__(self, key: str) -> float:
        return self.values.get(key, 0.0)


class BehaviorExtractor:
    """Computes behavioral features from programs with z-score normalization."""

    BUILT_IN_FEATURES: dict[str, Callable[[Program, ast.AST], float]] = {
        "code_length": compute_code_length,
        "ast_depth": compute_ast_depth,
        "cyclomatic_complexity": compute_cyclomatic_complexity,
        "loop_count": compute_loop_count,
        "math_operators": compute_math_operators,
        "branch_count": compute_branch_count,
        "loop_nesting_max": compute_loop_nesting_max,
        "function_def_count": compute_function_def_count,
        "numeric_literal_count": compute_numeric_literal_count,
        "comparison_count": compute_comparison_count,
        "subscript_count": compute_subscript_count,
        "call_count": compute_call_count,
        "comprehension_count": compute_comprehension_count,
        "range_max_arg": compute_range_max_arg,
    }

    def __init__(
        self,
        ast_features: Optional[list[str]] = None,
        score_keys: Optional[list[str]] = None,
        custom_extractors: Optional[dict[str, Callable[[Program], float]]] = None,
    ) -> None:
        if ast_features is None:
            self.ast_features = [
                "math_operators",
                "loop_nesting_max",
                "comprehension_count",
                "range_max_arg",
            ]
        else:
            self.ast_features = list(ast_features)

        if score_keys is None:
            self.score_keys = []
        else:
            self.score_keys = list(score_keys)

        # Custom extractors take (Program,) only — no AST dependency.
        # This allows non-code content types (e.g. prompts) to provide extractors.
        self._custom_extractors: dict[str, Callable[[Program], float]] = custom_extractors or {}
        self.extractors = {**self.BUILT_IN_FEATURES}

        # All features = AST features + score keys
        self.features = self.ast_features + self.score_keys

        # Welford's online algorithm for z-score normalization (adaptive mode)
        self._count: dict[str, int] = {f: 0 for f in self.features}
        self._mean: dict[str, float] = {f: 0.0 for f in self.features}
        self._M2: dict[str, float] = {f: 0.0 for f in self.features}

        # Fixed bounds mode (deterministic normalization)
        self._fixed_bounds: Optional[dict[str, tuple[float, float]]] = None

    def set_fixed_bounds(self, bounds: dict[str, tuple[float, float]]) -> None:
        """
        Set fixed bounds for deterministic normalization.

        When fixed bounds are set, normalization uses simple min-max scaling
        instead of adaptive z-score normalization. This ensures the same code
        always maps to the same behavior vector.

        Args:
            bounds: Dict mapping feature name to (min, max) tuple.
                    Raw values are normalized as: (value - min) / (max - min)
                    Values outside bounds are clipped to [0, 1].
        """
        self._fixed_bounds = {}
        for feature in self.features:
            if feature in bounds:
                lo, hi = bounds[feature]
                if hi <= lo:
                    raise ValueError(f"Invalid bounds for {feature}: max ({hi}) must be > min ({lo})")
                self._fixed_bounds[feature] = (float(lo), float(hi))
            else:
                # Default bounds for unknown features
                self._fixed_bounds[feature] = (0.0, 100.0)

    def has_fixed_bounds(self) -> bool:
        """Check if fixed bounds mode is enabled."""
        return self._fixed_bounds is not None

    def init_stats_from_data(self, feature_data: dict[str, list[float]]) -> None:
        """Initialize running stats from provided raw feature values."""
        for feature, values in feature_data.items():
            if feature not in self.features:
                continue
            n = len(values)
            if n == 0:
                continue
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
            self._count[feature] = n
            self._mean[feature] = mean
            self._M2[feature] = variance * (n - 1)

    def _update_stats(self, feature: str, value: float) -> None:
        """Update running mean and variance using Welford's algorithm."""
        self._count[feature] += 1
        delta = value - self._mean[feature]
        self._mean[feature] += delta / self._count[feature]
        delta2 = value - self._mean[feature]
        self._M2[feature] += delta * delta2

    def _get_std(self, feature: str) -> float:
        if self._count[feature] < 2:
            return 1.0
        variance = self._M2[feature] / (self._count[feature] - 1)
        return max(math.sqrt(variance), 0.1)

    def _zscore_to_01(self, z: float) -> float:
        """Convert z-score to [0, 1] using sigmoid."""
        z = max(-10, min(10, z))
        return 1.0 / (1.0 + math.exp(-z))

    def extract(self, program: Program, eval_result: Optional[dict] = None) -> FeatureVector:
        """Extract behavioral features from a program."""
        tree = None
        needs_ast = any(
            feature_name not in self._custom_extractors and feature_name in self.extractors
            for feature_name in self.ast_features
        )
        if needs_ast:
            try:
                tree = ast.parse(program.content)
            except SyntaxError:
                return FeatureVector({f: 0.5 for f in self.features})

        raw_values: dict[str, float] = {}

        # Extract AST-based features (built-in take (Program, AST))
        for feature_name in self.ast_features:
            if feature_name in self._custom_extractors:
                try:
                    raw_values[feature_name] = self._custom_extractors[feature_name](program)
                except Exception:
                    raw_values[feature_name] = 0.0
            elif feature_name in self.extractors:
                try:
                    raw_values[feature_name] = self.extractors[feature_name](program, tree)
                except Exception:
                    raw_values[feature_name] = 0.0
            else:
                raw_values[feature_name] = 0.0

        # Extract score-based features from eval result
        if eval_result:
            for key in self.score_keys:
                raw_values[key] = float(eval_result.get(key, 0.0))
        else:
            for key in self.score_keys:
                raw_values[key] = 0.0

        # Normalize features
        values: dict[str, float] = {}

        if self._fixed_bounds is not None:
            # Deterministic mode: use fixed min-max bounds
            for feature in self.features:
                raw = raw_values.get(feature, 0.0)
                lo, hi = self._fixed_bounds[feature]
                normalized = (raw - lo) / (hi - lo)
                values[feature] = float(np.clip(normalized, 0.0, 1.0))
        else:
            # Adaptive mode: z-score normalization with sigmoid
            for feature in self.features:
                raw = raw_values.get(feature, 0.0)
                self._update_stats(feature, raw)
                z = (raw - self._mean[feature]) / self._get_std(feature)
                values[feature] = self._zscore_to_01(z)

        return FeatureVector(values)
