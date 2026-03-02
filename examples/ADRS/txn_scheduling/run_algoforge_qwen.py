#!/usr/bin/env python3
"""Run AlgoForge for Transaction Scheduling using a local Qwen endpoint."""

from datetime import datetime
import json
from pathlib import Path

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, INPUTS, score_fn, DIVERSITY_SEED_PROMPT
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig,
    PunctuatedEquilibriumConfig, PromptOptConfig,
)

TXN_AST_FEATURES = ['loop_nesting_max', 'comparison_count', 'call_count', 'branch_count']

QWEN_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PROMPT_CACHE_PATH = Path("runs/20260224_191832_qwen/optimized_prompts.json")

if not PROMPT_CACHE_PATH.exists():
    raise FileNotFoundError(f"Prompt cache not found: {PROMPT_CACHE_PATH}")

prompt_data = json.loads(PROMPT_CACHE_PATH.read_text())
prompt_overrides = {
    "mutation": prompt_data.get("mutation", {}),
    "paradigm_shift": prompt_data.get("paradigm_shift"),
}

config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=INPUTS,
    score_fn=score_fn,
    paradigm_models=QWEN_MODEL,
    mutation_models=[QWEN_MODEL],
    local_endpoints={QWEN_MODEL: "http://localhost:8000/v1"},
    model_info={
        QWEN_MODEL: {
            "input_cost_per_token": 0.0000001,
            "output_cost_per_token": 0.0000004,
        },
    },
    budget=BudgetConfig(dollars=19.10),
    cvt=CVTConfig(n_centroids=50, defer_centroids=True),
    init=InitConfig(
        enabled=True,
        n_diverse_seeds=5,
        n_variants_per_seed=20,
        temperature=0.8,
        diversity_prompt=DIVERSITY_SEED_PROMPT,
    ),
    meta_advice=MetaAdviceConfig(enabled=True, interval=50),
    pipeline=PipelineConfig(n_llm_workers=12, n_eval_processes=12, n_inspirations=1, output_mode="full", eval_timeout=300),
    behavior=BehaviorConfig(ast_features=TXN_AST_FEATURES, score_keys=[], init_noise=0.3),
    punctuated_equilibrium=PunctuatedEquilibriumConfig(
        enabled=True,
        interval=5,
        n_clusters=3,
        n_variants=3,
        behavior_noise=0.3,
        temperature=0.7,
        reasoning_effort="low",
    ),
    prompt_opt=PromptOptConfig(enabled=False),
    prompt_overrides=prompt_overrides,
    output_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_qwen",
)

if __name__ == "__main__":
    run(config)
