"""
BehaviorExtractor: Computes behavioral features from programs.

Used by MAP-Elites style methods to characterize programs
along dimensions other than fitness.
"""

import ast
from dataclasses import dataclass
from typing import Callable, Optional

from ..core import Program
from .features import (
    compute_code_length,
    compute_ast_depth,
    compute_cyclomatic_complexity,
    compute_loop_count,
    compute_math_operators,
)


@dataclass
class FeatureVector:
    """Vector of behavioral features for a program."""

    values: dict[str, float]

    def to_array(self, feature_names: list[str]) -> list[float]:
        """Convert to array in specified feature order."""
        return [self.values.get(name, 0.0) for name in feature_names]

    def __getitem__(self, key: str) -> float:
        return self.values.get(key, 0.0)


class BehaviorExtractor:
    """
    Computes behavioral features from programs.

    Used by MAP-Elites style methods to characterize programs
    along dimensions other than fitness.
    """

    BUILT_IN_FEATURES: dict[str, Callable[[Program, ast.AST], float]] = {
        "code_length": compute_code_length,
        "ast_depth": compute_ast_depth,
        "cyclomatic_complexity": compute_cyclomatic_complexity,
        "loop_count": compute_loop_count,
        "math_operators": compute_math_operators,
    }

    def __init__(
        self,
        features: Optional[list[str]] = None,
        custom_extractors: Optional[dict[str, Callable[[Program, ast.AST], float]]] = None
    ) -> None:
        """
        Args:
            features: List of feature names to extract (defaults to all built-in)
            custom_extractors: Custom feature extractors {name: func}
        """
        self.features = features or list(self.BUILT_IN_FEATURES.keys())
        self.extractors = {**self.BUILT_IN_FEATURES, **(custom_extractors or {})}

    def extract(self, program: Program) -> FeatureVector:
        """Extract behavioral features from a program."""
        try:
            tree = ast.parse(program.code)
        except SyntaxError:
            return FeatureVector({f: 0.0 for f in self.features})

        values = {}
        for feature_name in self.features:
            if feature_name in self.extractors:
                try:
                    values[feature_name] = self.extractors[feature_name](program, tree)
                except Exception:
                    values[feature_name] = 0.0
            else:
                values[feature_name] = 0.0

        return FeatureVector(values)
