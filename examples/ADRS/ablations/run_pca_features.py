#!/usr/bin/env python3
"""
Ablation: PCA-selected features for TXN scheduling
====================================================

Runs TXN with the 4 PCA-selected best features (high variance, low correlation)
instead of the default 4 features.

Two conditions:
  pca_noise    CVT data-driven + noise=0.2  (with noise)
  pca_no_noise CVT data-driven + noise=0.0  (without noise)

3 seeds × 750 evals each.
"""

import sys
import json
import argparse
from pathlib import Path

from algoforge import (
    run, AlgoforgeConfig, BudgetConfig,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig,
    PunctuatedEquilibriumConfig, PromptOptConfig,
)

# ANOVA-selected features (high between-group / within-group F-statistic, low correlation)
ANOVA_FEATURES = ['subscript_count', 'ast_depth', 'loop_nesting_max', 'comprehension_count']

QWEN_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
N_SEEDS = 3
BUDGET_EVALS = 750

CONDITIONS = {
    "pca_noise": {
        "desc": f"PCA features {ANOVA_FEATURES} + CVT data-driven + noise=0.2",
        "cvt": CVTConfig(n_centroids=50, defer_centroids=True),
        "init_noise": 0.2,
    },
    "pca_no_noise": {
        "desc": f"PCA features {ANOVA_FEATURES} + CVT data-driven + noise=0.0",
        "cvt": CVTConfig(n_centroids=50, defer_centroids=True),
        "init_noise": 0.0,
    },
}


def make_config(condition_name: str, seed: int) -> AlgoforgeConfig:
    import problem
    cond = CONDITIONS[condition_name]
    output_dir = f"runs/ablations/anova_features/txn/{condition_name}_seed{seed}"

    return AlgoforgeConfig(
        problem_description=problem.PROBLEM_DESCRIPTION,
        function_signature=problem.FUNCTION_SIGNATURE,
        seed_program=problem.SEED_PROGRAM,
        inputs=problem.INPUTS,
        score_fn=problem.score_fn,
        paradigm_models=QWEN_MODEL,
        mutation_models=[QWEN_MODEL],
        local_endpoints={QWEN_MODEL: "http://localhost:8001/v1"},
        model_info={
            QWEN_MODEL: {
                "input_cost_per_token": 0.0000001,
                "output_cost_per_token": 0.0000004,
            },
        },
        budget=BudgetConfig(evaluations=BUDGET_EVALS),
        cvt=cond["cvt"],
        init=InitConfig(
            enabled=True,
            n_diverse_seeds=3,
            n_variants_per_seed=6,
            temperature=0.8,
            diversity_prompt=problem.DIVERSITY_SEED_PROMPT,
        ),
        meta_advice=MetaAdviceConfig(enabled=False),
        pipeline=PipelineConfig(
            n_llm_workers=4,
            n_eval_processes=4,
            n_inspirations=1,
            output_mode="full",
            eval_timeout=180.0,
            max_tokens=8192,
        ),
        behavior=BehaviorConfig(
            ast_features=ANOVA_FEATURES,
            score_keys=[],
            init_noise=cond["init_noise"],
        ),
        punctuated_equilibrium=PunctuatedEquilibriumConfig(enabled=False),
        prompt_opt=PromptOptConfig(enabled=False),
        output_dir=output_dir,
    )


def run_condition(condition_name: str, n_seeds: int = N_SEEDS):
    cond = CONDITIONS[condition_name]
    print(f"\n{'='*65}")
    print(f"Problem: txn  |  Condition: {condition_name}")
    print(f"  {cond['desc']}")
    print(f"  {n_seeds} seeds x {BUDGET_EVALS} evals")
    print(f"{'='*65}")

    results = []
    for seed in range(n_seeds):
        print(f"\n  Seed {seed}/{n_seeds-1}...")
        config = make_config(condition_name, seed)

        try:
            result = run(config)
            entry = {
                "seed": seed,
                "best_score": result.best_score,
                "total_evaluations": result.total_evaluations,
                "runtime_seconds": result.runtime_seconds,
                "archive_size": result.archive_size,
                "output_dir": config.output_dir,
            }
            results.append(entry)
            print(f"  > Seed {seed}: best={result.best_score:.2f}, "
                  f"evals={result.total_evaluations}, "
                  f"archive={result.archive_size}")
        except Exception as e:
            print(f"  X Seed {seed} FAILED: {e}")
            results.append({"seed": seed, "error": str(e)})

    out_dir = Path(f"runs/ablations/anova_features/txn")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{condition_name}_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Summary -> {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=list(CONDITIONS.keys()))
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    # TXN scheduling on sys.path
    prob_dir = Path(__file__).parent.parent / "txn_scheduling"
    sys.path.insert(0, str(prob_dir))

    run_condition(args.condition, args.seeds)
