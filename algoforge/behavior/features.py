"""
Built-in feature computation functions.

These functions compute structural features from program code,
used for behavioral diversity in MAP-Elites style methods.
"""

import ast
from typing import Optional

from ..core import Program


def compute_code_length(program: Program, tree: Optional[ast.AST] = None) -> float:
    """Character count of code."""
    return float(len(program.code))


def compute_ast_depth(program: Program, tree: ast.AST) -> float:
    """Maximum depth of AST."""
    def _depth(node: ast.AST) -> int:
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 1
        return 1 + max(_depth(c) for c in children)
    return float(_depth(tree))


def compute_cyclomatic_complexity(program: Program, tree: ast.AST) -> float:
    """McCabe cyclomatic complexity."""
    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
    return float(complexity)


def compute_loop_count(program: Program, tree: ast.AST) -> float:
    """Count of loop constructs."""
    count = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
    return float(count)


def compute_math_operators(program: Program, tree: ast.AST) -> float:
    """Count of mathematical operators in AST (BinOp, UnaryOp)."""
    def _count(node: ast.AST) -> int:
        count = 1 if isinstance(node, (ast.BinOp, ast.UnaryOp)) else 0
        for child in ast.iter_child_nodes(node):
            count += _count(child)
        return count
    return float(_count(tree))
