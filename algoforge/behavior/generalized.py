"""
GeneralizedBehaviorExtractor: Domain-agnostic behavior features for evolutionary code search.

Works across problem types without domain-specific knowledge.
"""

import ast
from typing import Optional

from ..core import Program
from .extractor import FeatureVector


class GeneralizedBehaviorExtractor:
    """
    Domain-agnostic behavior features for evolutionary code search.
    Works across problem types without domain-specific knowledge.
    """

    # Configurable feature sets
    FEATURE_SETS = {
        'minimal': ['execution_time', 'primary_score', 'loop_count'],
        'standard': [
            'execution_time', 'primary_score',
            'loop_count', 'branch_count', 'function_count',
            'loop_nesting_max', 'early_exit_count',
        ],
        'full': [
            'execution_time', 'primary_score',
            'loop_count', 'branch_count', 'function_count',
            'loop_nesting_max', 'early_exit_count', 'recursion_detected',
            'comprehension_count', 'variable_count', 'external_call_count',
        ],
    }

    def __init__(
        self,
        feature_set: str = 'standard',
        custom_features: Optional[list[str]] = None,
        time_key: str = 'execution_time',
        score_key: str = 'primary_score',
        max_time: float = 60.0,
        max_score: float = 100.0,
    ):
        """
        Args:
            feature_set: One of 'minimal', 'standard', 'full'
            custom_features: Override with specific feature list
            time_key: Metadata key for execution time
            score_key: Metadata key for primary score
            max_time: Max time for normalization (time -> 0, 0 -> 1)
            max_score: Max score for normalization (0 -> 0, max -> 1)
        """
        if custom_features:
            self.features = custom_features
        else:
            self.features = self.FEATURE_SETS.get(feature_set, self.FEATURE_SETS['standard'])

        self.time_key = time_key
        self.score_key = score_key
        self.max_time = max_time
        self.max_score = max_score

    def extract(self, program: Program) -> FeatureVector:
        """Extract generalized behavioral features."""
        code = program.code
        metadata = program.metadata or {}
        values = {}

        # Parse AST once
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return FeatureVector({f: 0.0 for f in self.features})

        # === Dynamic features (from evaluation metadata) ===
        if 'execution_time' in self.features:
            # Normalize: max_time -> 0, 0 -> 1
            raw_time = metadata.get(self.time_key, self.max_time)
            values['execution_time'] = max(0.0, 1.0 - raw_time / self.max_time)

        if 'primary_score' in self.features:
            # Normalize to 0-1 range
            raw_score = metadata.get(self.score_key, 0.0)
            values['primary_score'] = raw_score / self.max_score if self.max_score > 0 else 0.0

        # === Static features (from code analysis) ===
        if 'loop_count' in self.features:
            values['loop_count'] = float(sum(
                1 for node in ast.walk(tree)
                if isinstance(node, (ast.For, ast.While))
            ))

        if 'branch_count' in self.features:
            if_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.If))
            ternary_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.IfExp))
            values['branch_count'] = float(if_count + ternary_count)

        if 'function_count' in self.features:
            # Exclude the main function (count helpers only)
            func_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            values['function_count'] = float(max(0, func_count - 1))

        if 'loop_nesting_max' in self.features:
            values['loop_nesting_max'] = float(self._max_loop_nesting(tree))

        if 'early_exit_count' in self.features:
            values['early_exit_count'] = float(self._count_early_exits(tree))

        if 'recursion_detected' in self.features:
            values['recursion_detected'] = float(self._detect_recursion(tree))

        if 'comprehension_count' in self.features:
            values['comprehension_count'] = float(sum(
                1 for node in ast.walk(tree)
                if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
            ))

        if 'variable_count' in self.features:
            values['variable_count'] = float(self._count_variables(tree))

        if 'external_call_count' in self.features:
            values['external_call_count'] = float(self._count_external_calls(tree))

        return FeatureVector(values)

    def _max_loop_nesting(self, tree: ast.AST) -> int:
        """Find maximum loop nesting depth."""
        def _depth(node: ast.AST, current: int) -> int:
            max_depth = current
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While)):
                    max_depth = max(max_depth, _depth(child, current + 1))
                else:
                    max_depth = max(max_depth, _depth(child, current))
            return max_depth
        return _depth(tree, 0)

    def _count_early_exits(self, tree: ast.AST) -> int:
        """Count return/break statements inside loops."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, (ast.Return, ast.Break)):
                        count += 1
        return count

    def _detect_recursion(self, tree: ast.AST) -> int:
        """Detect if any function calls itself."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name) and child.func.id == func_name:
                            return 1
        return 0

    def _count_variables(self, tree: ast.AST) -> int:
        """Count unique variable assignments."""
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    names.add(node.target.id)
        return len(names)

    def _count_external_calls(self, tree: ast.AST) -> int:
        """Count calls to non-local functions (library/module calls)."""
        count = 0
        local_funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

        # Get builtins safely
        import builtins
        builtin_names = set(dir(builtins))

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # method call like torch.zeros() or obj.method()
                    count += 1
                elif isinstance(node.func, ast.Name):
                    name = node.func.id
                    if name not in local_funcs and name not in builtin_names:
                        count += 1
        return count
