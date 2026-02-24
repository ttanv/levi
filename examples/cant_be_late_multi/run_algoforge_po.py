#!/usr/bin/env python3
"""Run AlgoForge with Prompt Optimization for Can't Be Late Multi-Region Scheduling."""

import ast
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add algoforge to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, INPUTS, score_fn, DIVERSITY_SEED_PROMPT
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig, SamplerModelPair,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig,
    PunctuatedEquilibriumConfig, PromptOptConfig,
)
from algoforge.core import Program


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


# All 5 behavioral dimensions: 3 domain-specific + 2 AST
CANT_BE_LATE_MULTI_AST_FEATURES = [
    'state_tracking_level',
    'math_model_complexity',
    'threshold_count',
    'loop_count',
    'decision_branch_depth',
]

CUSTOM_EXTRACTORS = {
    'state_tracking_level': _compute_state_tracking_level,
    'math_model_complexity': _compute_math_model_complexity,
    'threshold_count': _compute_threshold_count,
    'decision_branch_depth': _compute_decision_branch_depth,
}

# Models
LIGHT_MODELS = [
    "openrouter/xiaomi/mimo-v2-flash",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
]
PARADIGM_SHIFT_MODEL = "openrouter/google/gemini-3-flash-preview"

LOCAL_ENDPOINTS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8001/v1",
}

MODEL_INFO = {
    "xiaomi/mimo-v2-flash": {
        "max_tokens": 16384,
        "max_input_tokens": 262144,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.00000009,
        "output_cost_per_token": 0.00000029,
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000004,
    },
    "google/gemini-3-flash-preview": {
        "max_tokens": 65536,
        "max_input_tokens": 1048576,
        "max_output_tokens": 65536,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.000003,
    },
}

run_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_po"

config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=INPUTS,
    score_fn=score_fn,
    budget=BudgetConfig(dollars=5.00),
    sampler_model_pairs=[
        # MiMo-V2-Flash (OpenRouter)
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=0.3),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=0.7),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=1.0),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=1.2),
        # Qwen 30B (Local TPU)
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=0.3),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=0.7),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=1.0),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=1.2),
        # Gemini Flash (OpenRouter) - paradigm shift model
        SamplerModelPair(sampler="softmax", model=PARADIGM_SHIFT_MODEL, weight=1.0, temperature=0.3),
    ],
    cvt=CVTConfig(n_centroids=50, defer_centroids=True),
    init=InitConfig(
        enabled=True,
        n_diverse_seeds=5,
        n_variants_per_seed=20,
        diversity_model=PARADIGM_SHIFT_MODEL,
        variant_models=LIGHT_MODELS,
        temperature=0.8,
        diversity_prompt=DIVERSITY_SEED_PROMPT,
    ),
    meta_advice=MetaAdviceConfig(enabled=True, interval=50, model=PARADIGM_SHIFT_MODEL),
    pipeline=PipelineConfig(n_llm_workers=12, n_eval_processes=12, n_inspirations=1, output_mode="full", eval_timeout=300),
    behavior=BehaviorConfig(
        ast_features=CANT_BE_LATE_MULTI_AST_FEATURES,
        score_keys=[],
        init_noise=0.3,
        custom_extractors=CUSTOM_EXTRACTORS,
    ),
    punctuated_equilibrium=PunctuatedEquilibriumConfig(
        enabled=True,
        interval=5,
        n_clusters=3,
        n_variants=3,
        heavy_models=[PARADIGM_SHIFT_MODEL],
        variant_models=LIGHT_MODELS,
        behavior_noise=0.3,
        temperature=0.7,
        reasoning_effort="low",
    ),
    prompt_opt=PromptOptConfig(enabled=True),
    output_dir=run_dir,
    local_endpoints=LOCAL_ENDPOINTS,
    model_info=MODEL_INFO,
)

if __name__ == "__main__":
    run(config)
