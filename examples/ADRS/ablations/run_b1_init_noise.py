#!/usr/bin/env python3
"""
Ablation B1: Centroid Initialization Noise
===========================================
Tests whether the Gaussian noise added to behavior vectors during init matters.

The noise spreads init programs across cells, preventing early lock-in and
ensuring diverse centroid coverage from the start.

Conditions:
  no_noise    - init_noise=0.0 (pure k-means on actual positions)
  low_noise   - init_noise=0.15 (default BehaviorConfig value)
  high_noise  - init_noise=0.30 (current txn_scheduling setting)

Usage:
  python run_b1_init_noise.py
  python run_b1_init_noise.py --condition no_noise --seeds 1
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "txn_scheduling"))

from problem import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, INPUTS, score_fn, DIVERSITY_SEED_PROMPT
from algoforge import (
    run, AlgoforgeConfig, BudgetConfig,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig,
    PunctuatedEquilibriumConfig, PromptOptConfig,
)

QWEN_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
TXN_AST_FEATURES = ['loop_nesting_max', 'comparison_count', 'call_count', 'branch_count']

N_SEEDS = 3
BUDGET_EVALS = 200
N_LLM_WORKERS = 8
N_EVAL_WORKERS = 8
EVAL_TIMEOUT = 60.0

INIT_N_SEEDS = 3
INIT_N_VARIANTS = 8

CONDITIONS = {
    "no_noise": {
        "desc": "No init noise (init_noise=0.0) — programs map to exact AST positions",
        "init_noise": 0.0,
    },
    "low_noise": {
        "desc": "Low init noise (init_noise=0.15) — default BehaviorConfig",
        "init_noise": 0.15,
    },
    "high_noise": {
        "desc": "High init noise (init_noise=0.30) — current txn_scheduling setting",
        "init_noise": 0.30,
    },
}


def make_config(condition_name: str, seed: int) -> AlgoforgeConfig:
    cond = CONDITIONS[condition_name]
    output_dir = f"runs/ablations/b1/{condition_name}_seed{seed}"

    return AlgoforgeConfig(
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
        budget=BudgetConfig(evaluations=BUDGET_EVALS),
        cvt=CVTConfig(n_centroids=50, defer_centroids=True),   # Fixed: LEVI archive
        init=InitConfig(
            enabled=True,
            n_diverse_seeds=INIT_N_SEEDS,
            n_variants_per_seed=INIT_N_VARIANTS,
            temperature=0.8,
            diversity_prompt=DIVERSITY_SEED_PROMPT,
        ),
        meta_advice=MetaAdviceConfig(enabled=False),
        pipeline=PipelineConfig(
            n_llm_workers=N_LLM_WORKERS,
            n_eval_processes=N_EVAL_WORKERS,
            n_inspirations=1,
            output_mode="full",
            eval_timeout=EVAL_TIMEOUT,
            max_tokens=8192,
        ),
        behavior=BehaviorConfig(
            ast_features=TXN_AST_FEATURES,
            score_keys=[],
            init_noise=cond["init_noise"],   # <-- the variable
        ),
        punctuated_equilibrium=PunctuatedEquilibriumConfig(enabled=False),
        prompt_opt=PromptOptConfig(enabled=False),
        output_dir=output_dir,
    )


def run_ablation(conditions=None, n_seeds=None):
    if conditions is None:
        conditions = list(CONDITIONS.keys())
    if n_seeds is None:
        n_seeds = N_SEEDS

    summary = {}

    for cond_name in conditions:
        print(f"\n{'='*60}")
        print(f"B1 | Condition: {cond_name}")
        print(f"   {CONDITIONS[cond_name]['desc']}")
        print(f"{'='*60}")

        cond_results = []
        for seed in range(n_seeds):
            print(f"\n  Seed {seed}/{n_seeds-1}...")
            config = make_config(cond_name, seed)

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
                cond_results.append(entry)
                print(f"  ✓ Seed {seed}: best={result.best_score:.2f}, "
                      f"evals={result.total_evaluations}, "
                      f"archive={result.archive_size}, "
                      f"time={result.runtime_seconds:.0f}s")
            except Exception as e:
                print(f"  ✗ Seed {seed} FAILED: {e}")
                cond_results.append({"seed": seed, "error": str(e)})

        summary[cond_name] = cond_results

    out_dir = Path("runs/ablations/b1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B1: Init noise ablation")
    parser.add_argument(
        "--condition", choices=list(CONDITIONS.keys()),
        help="Run only this condition (default: all)"
    )
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS,
        help=f"Number of seeds per condition (default: {N_SEEDS})"
    )
    args = parser.parse_args()

    conditions = [args.condition] if args.condition else None
    run_ablation(conditions, args.seeds)
