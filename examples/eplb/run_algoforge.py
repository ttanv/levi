#!/usr/bin/env python3
"""Run AlgoForge on EPLB (Expert Parallelism Load Balancer) Problem."""

from datetime import datetime

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, get_lazy_inputs, score_fn
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig, SamplerModelPair,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig
)
from algoforge.config.models import PunctuatedEquilibriumConfig

# Behavioral dimensions: 3 code structure + 4 performance profile = 7 total
EPLB_AST_FEATURES = ['loop_nesting_max', 'cyclomatic_complexity', 'math_operators']
EPLB_SCORE_KEYS = ['execution_time', 'workload_main', 'workload_8', 'workload_9']

# --- Config ---
# Local TPU models (hosted_vllm/ prefix for OpenAI-compatible vLLM endpoints)
LIGHT_MODELS = [
    "hosted_vllm/Qwen/Qwen3-30B-A3B-Instruct-2507",
    "hosted_vllm/google/gemma-3-27b-it",
]
HEAVY_MODEL = "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"

# Model -> API base URL mapping for local TPU endpoints
API_BASES = {
    "hosted_vllm/Qwen/Qwen3-30B-A3B-Instruct-2507": "http://10.142.0.3:8000/v1",
    "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct": "http://10.130.0.4:8000/v1",
    "hosted_vllm/google/gemma-3-27b-it": "http://10.164.0.4:8000/v1",
}

RUN_DIR = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

config = AlgoforgeConfig(
    problem_description=PROBLEM_DESCRIPTION,
    function_signature=FUNCTION_SIGNATURE,
    seed_program=SEED_PROGRAM,
    inputs=get_lazy_inputs(),
    score_fn=score_fn,
    budget=BudgetConfig(dollars=3.0),
    sampler_model_pairs=[
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=0.7),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=0.7),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[0], weight=1.0, temperature=1.2),
        SamplerModelPair(sampler="softmax", model=LIGHT_MODELS[1], weight=1.0, temperature=1.2),
    ],
    cvt=CVTConfig(n_centroids=40, defer_centroids=True),
    init=InitConfig(
        n_diverse_seeds=5,
        n_variants_per_seed=50,
        diversity_model=HEAVY_MODEL,
        variant_models=LIGHT_MODELS,  # Cycle through all light models
    ),
    meta_advice=MetaAdviceConfig(interval=50, model=HEAVY_MODEL),
    pipeline=PipelineConfig(n_llm_workers=8, n_eval_processes=8, n_inspirations=1, output_mode="diff"),
    behavior=BehaviorConfig(ast_features=EPLB_AST_FEATURES, score_keys=EPLB_SCORE_KEYS),
    punctuated_equilibrium=PunctuatedEquilibriumConfig(
        enabled=True,
        interval=5,
        n_clusters=3,
        n_variants=1,
        heavy_model=HEAVY_MODEL,
        variant_models=LIGHT_MODELS,
        behavior_noise=0.3,
        temperature=1.0,
    ),
    output_dir=RUN_DIR,
    api_bases=API_BASES,
)

# --- Run ---
if __name__ == "__main__":
    run(config)
