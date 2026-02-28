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
    return float(len(program.content))


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


def compute_branch_count(program: Program, tree: ast.AST) -> float:
    """Count of if/elif/else branches."""
    count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.If))
    return float(count)


def compute_loop_nesting_max(program: Program, tree: ast.AST) -> float:
    """Maximum depth of nested loops."""
    def _depth(node: ast.AST, current: int) -> int:
        max_depth = current
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                max_depth = max(max_depth, _depth(child, current + 1))
            else:
                max_depth = max(max_depth, _depth(child, current))
        return max_depth
    return float(_depth(tree, 0))


def compute_function_def_count(program: Program, tree: ast.AST) -> float:
    """Count of function definitions (captures modularity)."""
    count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    return float(count)


def compute_numeric_literal_count(program: Program, tree: ast.AST) -> float:
    """Count of numeric literals (captures parameter tuning density)."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            count += 1
        # Python 3.7 compatibility: ast.Num was deprecated but may still appear
        elif isinstance(node, ast.Num):
            count += 1
    return float(count)


def compute_comparison_count(program: Program, tree: ast.AST) -> float:
    """Count of comparison operations (captures decision complexity)."""
    count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Compare))
    return float(count)


def compute_subscript_count(program: Program, tree: ast.AST) -> float:
    """Count of subscript/indexing operations (captures data structure manipulation)."""
    count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Subscript))
    return float(count)


def compute_call_count(program: Program, tree: ast.AST) -> float:
    """Count of function calls (captures call density)."""
    count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call))
    return float(count)


def compute_comprehension_count(program: Program, tree: ast.AST) -> float:
    """Count of list/dict/set comprehensions and generator expressions."""
    count = sum(
        1 for node in ast.walk(tree)
        if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
    )
    return float(count)


def compute_range_max_arg(program: Program, tree: ast.AST) -> float:
    """Maximum argument passed to range() calls (captures iteration bounds)."""
    max_val = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check if it's a call to range()
            func = node.func
            if isinstance(func, ast.Name) and func.id == 'range':
                # Extract numeric arguments from range()
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                        max_val = max(max_val, abs(arg.value))
                    elif isinstance(arg, ast.Num):  # Python 3.7 compatibility
                        max_val = max(max_val, abs(arg.n))
    return float(max_val)
