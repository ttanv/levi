#!/usr/bin/env python3
"""
Ablation A1: Archive Type Ablation
===================================
Tests whether CVT-MAP-Elites (LEVI's archive) is what prevents convergence/plateau.

Conditions:
  levi_cvt    - CVT-MAP-Elites, data-driven centroids (LEVI full)
  elitist     - Single centroid (keep only best = elitist selection)
  uniform_cvt - CVT-MAP-Elites, uniform-random centroids (not data-driven)

Usage:
  python run_a1_archive.py                     # All conditions, 3 seeds each
  python run_a1_archive.py --condition levi_cvt
  python run_a1_archive.py --condition elitist --seeds 1
"""

import sys
import json
import argparse
from pathlib import Path

# Make txn_scheduling importable
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
BUDGET_EVALS = 200       # Total evals incl. init phase
N_LLM_WORKERS = 8
N_EVAL_WORKERS = 8
EVAL_TIMEOUT = 60.0      # seconds per eval

# Init: 3 diverse seeds × 8 variants = ~24 init programs
# This leaves ~176 evals for evolution
INIT_N_SEEDS = 3
INIT_N_VARIANTS = 8

CONDITIONS = {
    "levi_cvt": {
        "desc": "LEVI full (CVT, data-driven centroids, 50 cells)",
        "cvt": CVTConfig(n_centroids=50, defer_centroids=True),
    },
    "elitist": {
        "desc": "Elitist (1 cell = keep only global best)",
        "cvt": CVTConfig(n_centroids=1, defer_centroids=False),
    },
    "uniform_cvt": {
        "desc": "Uniform CVT (50 cells, uniform-random centroids, not data-driven)",
        "cvt": CVTConfig(n_centroids=50, defer_centroids=False),
    },
}


def make_config(condition_name: str, seed: int) -> AlgoforgeConfig:
    cond = CONDITIONS[condition_name]
    output_dir = f"runs/ablations/a1/{condition_name}_seed{seed}"

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
        cvt=cond["cvt"],
        init=InitConfig(
            enabled=True,
            n_diverse_seeds=INIT_N_SEEDS,
            n_variants_per_seed=INIT_N_VARIANTS,
            temperature=0.8,
            diversity_prompt=DIVERSITY_SEED_PROMPT,
        ),
        meta_advice=MetaAdviceConfig(enabled=False),   # Disabled: isolate archive effect
        pipeline=PipelineConfig(
            n_llm_workers=N_LLM_WORKERS,
            n_eval_processes=N_EVAL_WORKERS,
            n_inspirations=1,
            output_mode="full",
            eval_timeout=EVAL_TIMEOUT,
            max_tokens=8192,   # 16k ctx - ~2.5k input = ~13.5k headroom; 8192 is safe
        ),
        behavior=BehaviorConfig(
            ast_features=TXN_AST_FEATURES,
            score_keys=[],
            init_noise=0.2,
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
        print(f"A1 | Condition: {cond_name}")
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

    # Save summary
    out_dir = Path("runs/ablations/a1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A1: Archive type ablation")
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
