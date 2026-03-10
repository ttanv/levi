#!/usr/bin/env python3
"""Run Levi for Can't Be Late Multi-Region Scheduling."""

import ast
import re
from datetime import datetime
from typing import Optional

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, INPUTS, score_fn, DIVERSITY_SEED_PROMPT
import levi
from levi.core import Program


# --- Domain-specific behavioral feature extractors ---
# These capture algorithmic *family* differences rather than code style/verbosity.
_STATE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'hasattr\s*\(', r'getattr\s*\(', r'setattr\s*\(',
        r'ctx\.\w+\s*=',
        r'_state', r'_last_', r'_prev_',
        r'cooldown', r'history', r'counter', r'_count',
        r'staleness', r'last_check', r'last_region', r'_tracker',
    ]
]

def _compute_state_tracking_level(program: Program, tree: Optional[ast.AST] = None) -> float:
    """Count stateful patterns (hasattr/setattr, ctx mutation, cooldowns, history tracking).

    Separates heavy-stateful strategies (region cooldown maps, preemption history)
    from stateless reactive strategies. Orthogonal to all AST complexity metrics.
    """
    return float(sum(len(p.findall(program.code)) for p in _STATE_PATTERNS))


_MATH_MODEL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'math\.exp', r'math\.log', r'math\.pow', r'math\.sqrt',
        r'math\.ceil', r'math\.floor',
        r'\*\*\s*[0-9]',
        r'exp\s*\(', r'probability', r'risk_score', r'decay', r'sigmoid',
    ]
]

def _compute_math_model_complexity(program: Program, tree: Optional[ast.AST] = None) -> float:
    """Count probabilistic/statistical modeling constructs (exp, log, decay, probability).

    Separates probabilistic strategies (stochastic aggression, temporal arbitrage)
    from deterministic if/else strategies. Orthogonal to state_tracking_level (r=-0.13).
    """
    return float(sum(len(p.findall(program.code)) for p in _MATH_MODEL_PATTERNS))


_THRESHOLD_PATTERN = re.compile(r'(?<![a-zA-Z_])0\.\d+')

def _compute_threshold_count(program: Program, tree: Optional[ast.AST] = None) -> float:
    """Count hardcoded float thresholds (0.xx patterns).

    Separates parameter-tuned strategies (many hand-tuned cutoffs for different
    scenarios) from clean analytical strategies. Orthogonal to both state_tracking
    (r=-0.15) and math_model (r=0.17).
    """
    return float(len(_THRESHOLD_PATTERN.findall(program.code)))


def _compute_decision_branch_depth(program: Program, tree: ast.AST) -> float:
    """Max nesting depth of if-statements. Captures decision tree structure."""
    def _max_if_depth(node: ast.AST, depth: int = 0) -> int:
        max_d = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                max_d = max(max_d, _max_if_depth(child, depth + 1))
            else:
                max_d = max(max_d, _max_if_depth(child, depth))
        return max_d
    return float(_max_if_depth(tree))


CUSTOM_EXTRACTORS = {
    'state_tracking_level': _compute_state_tracking_level,
    'math_model_complexity': _compute_math_model_complexity,
    'threshold_count': _compute_threshold_count,
    'decision_branch_depth': _compute_decision_branch_depth,
}

result = levi.evolve_code(
    PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    score_fn=score_fn,
    inputs=INPUTS,
    paradigm_model="openrouter/google/gemini-3-flash-preview",
    mutation_model=[
        "openrouter/xiaomi/mimo-v2-flash",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ],
    budget_dollars=5.00,
    local_endpoints={"Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8001/v1"},
    model_info={
        "xiaomi/mimo-v2-flash": {
            "input_cost_per_token": 0.00000009,
            "output_cost_per_token": 0.00000029,
        },
        "Qwen/Qwen3-30B-A3B-Instruct-2507": {
            "input_cost_per_token": 0.0000001,
            "output_cost_per_token": 0.0000004,
        },
    },
    init=levi.InitConfig(
        n_diverse_seeds=5,
        n_variants_per_seed=20,
        diversity_prompt=DIVERSITY_SEED_PROMPT,
    ),
    pipeline=levi.PipelineConfig(n_llm_workers=12, n_eval_processes=12, n_inspirations=1, eval_timeout=300),
    behavior=levi.BehaviorConfig(
        ast_features=[
            'state_tracking_level',
            'math_model_complexity',
            'threshold_count',
            'loop_count',
            'decision_branch_depth',
        ],
        init_noise=0.3,
        custom_extractors=CUSTOM_EXTRACTORS,
    ),
    punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(
        enabled=True,
        interval=5,
        n_clusters=3,
        n_variants=3,
        behavior_noise=0.3,
        temperature=0.7,
        reasoning_effort="low",
    ),
    prompt_opt=levi.PromptOptConfig(enabled=True),
    output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_po",
)
