#!/usr/bin/env python3
"""
Combined Ablation: Archive Type × Init Strategy — Two Problems
==============================================================

4 conditions in a 2×2 factorial (archive × init noise):

  levi_full     CVT data-driven  + noise=0.2  (full LEVI)
  no_cvt        Elitist (1 cell) + noise=0.2  (no archive)
  uniform_init  CVT uniform      + noise=0.2  (archive, not data-driven)
  no_noise      CVT data-driven  + noise=0.0  (archive, no init noise)

Separates:
  levi_full vs no_cvt       → does ANY archive help?          (headline)
  levi_full vs uniform_init → does data-driven placement help?
  levi_full vs no_noise     → does init noise help?

Two problems:
  txn  Transaction Scheduling  (algo diversity; AST descriptors)
  cbl  Can't Be Late Scheduling (cloud instance scheduling; 1080 real traces)

Usage:
  # Run one combination (launch 8 at once across 4 conditions × 2 problems):
  python run_combined.py --problem txn --condition levi_full
  python run_combined.py --problem txn --condition no_cvt
  python run_combined.py --problem txn --condition uniform_init
  python run_combined.py --problem txn --condition no_noise
  python run_combined.py --problem cbl --condition levi_full
  ...etc
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

# Maps --problem arg to example directory name.
# sys.path is set once at startup based on --problem (only one problem per process),
# so the regular `import problem` works and is pickle-safe for multiprocessing.
PROBLEM_DIRS = {
    "txn": "txn_scheduling",
    "cbl": "cant_be_late",
}

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
QWEN_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

N_SEEDS = 3
BUDGET_EVALS = 750
N_LLM_WORKERS = 4      # reduced per-run since conditions run in parallel
N_EVAL_WORKERS = 4
INIT_N_SEEDS = 3
INIT_N_VARIANTS = 6    # 3×6 = 18 init programs; fast init, plenty left for evolution

# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------
def _load_problem():
    """Import the `problem` module (already on sys.path via startup)."""
    import problem
    return problem

def _txn_problem():
    p = _load_problem()
    return dict(
        problem_description=p.PROBLEM_DESCRIPTION,
        function_signature=p.FUNCTION_SIGNATURE,
        seed_program=p.SEED_PROGRAM,
        inputs=p.INPUTS,
        score_fn=p.score_fn,
        diversity_prompt=p.DIVERSITY_SEED_PROMPT,
        ast_features=['loop_nesting_max', 'comparison_count', 'call_count', 'branch_count'],
        eval_timeout=180.0,
    )

def _cbl_problem():
    p = _load_problem()
    return dict(
        problem_description=p.PROBLEM_DESCRIPTION,
        function_signature=p.FUNCTION_SIGNATURE,
        seed_program=p.SEED_PROGRAM,
        inputs=p.INPUTS,
        score_fn=p.score_fn,
        diversity_prompt=p.DIVERSITY_SEED_PROMPT,
        ast_features=['loop_nesting_max', 'comparison_count', 'call_count', 'branch_count'],
        eval_timeout=60.0,
    )

PROBLEMS = {
    "txn": _txn_problem,
    "cbl": _cbl_problem,
}

# ---------------------------------------------------------------------------
# Condition definitions — vary ONLY archive type and init noise
# ---------------------------------------------------------------------------
CONDITIONS = {
    "levi_full": {
        "desc": "LEVI full: CVT data-driven centroids + init noise=0.2",
        "cvt": CVTConfig(n_centroids=50, defer_centroids=True),
        "init_noise": 0.2,
    },
    "no_cvt": {
        "desc": "No archive: elitist (1 cell) + init noise=0.2",
        "cvt": CVTConfig(n_centroids=1, defer_centroids=False),
        "init_noise": 0.2,
    },
    "uniform_init": {
        "desc": "Uniform CVT: random centroid placement + init noise=0.2",
        "cvt": CVTConfig(n_centroids=50, defer_centroids=False),
        "init_noise": 0.2,
    },
    "no_noise": {
        "desc": "LEVI no noise: CVT data-driven centroids + init noise=0.0",
        "cvt": CVTConfig(n_centroids=50, defer_centroids=True),
        "init_noise": 0.0,
    },
}


def make_config(problem_name: str, condition_name: str, seed: int) -> AlgoforgeConfig:
    prob = PROBLEMS[problem_name]()
    cond = CONDITIONS[condition_name]

    # Switch sys.path back to the right problem dir before importing
    output_dir = f"runs/ablations/combined_750/{problem_name}/{condition_name}_seed{seed}"

    return AlgoforgeConfig(
        problem_description=prob["problem_description"],
        function_signature=prob["function_signature"],
        seed_program=prob["seed_program"],
        inputs=prob["inputs"],
        score_fn=prob["score_fn"],
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
            n_diverse_seeds=INIT_N_SEEDS,
            n_variants_per_seed=INIT_N_VARIANTS,
            temperature=0.8,
            diversity_prompt=prob["diversity_prompt"],
        ),
        meta_advice=MetaAdviceConfig(enabled=False),
        pipeline=PipelineConfig(
            n_llm_workers=N_LLM_WORKERS,
            n_eval_processes=N_EVAL_WORKERS,
            n_inspirations=1,
            output_mode="full",
            eval_timeout=prob["eval_timeout"],
            max_tokens=8192,
        ),
        behavior=BehaviorConfig(
            ast_features=prob["ast_features"],
            score_keys=[],
            init_noise=cond["init_noise"],
        ),
        punctuated_equilibrium=PunctuatedEquilibriumConfig(enabled=False),
        prompt_opt=PromptOptConfig(enabled=False),
        output_dir=output_dir,
    )


def run_condition(problem_name: str, condition_name: str, n_seeds: int = N_SEEDS):
    cond = CONDITIONS[condition_name]
    print(f"\n{'='*65}")
    print(f"Problem: {problem_name}  |  Condition: {condition_name}")
    print(f"  {cond['desc']}")
    print(f"  {n_seeds} seeds × {BUDGET_EVALS} evals")
    print(f"{'='*65}")

    results = []
    for seed in range(n_seeds):
        print(f"\n  Seed {seed}/{n_seeds-1}...")
        config = make_config(problem_name, condition_name, seed)

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
            print(f"  ✓ Seed {seed}: best={result.best_score:.2f}, "
                  f"evals={result.total_evaluations}, "
                  f"archive={result.archive_size}")
        except Exception as e:
            print(f"  ✗ Seed {seed} FAILED: {e}")
            results.append({"seed": seed, "error": str(e)})

    # Save per-condition summary
    out_dir = Path(f"runs/ablations/combined_750/{problem_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{condition_name}_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Summary → {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined archive × init ablation on two problems"
    )
    parser.add_argument(
        "--problem", required=True, choices=list(PROBLEMS.keys()),
        help="Which problem to run"
    )
    parser.add_argument(
        "--condition", required=True, choices=list(CONDITIONS.keys()),
        help="Which condition to run"
    )
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS,
        help=f"Number of seeds (default: {N_SEEDS})"
    )
    args = parser.parse_args()

    # Put the correct problem directory on sys.path BEFORE anything imports `problem`.
    prob_dir = Path(__file__).parent.parent / PROBLEM_DIRS[args.problem]
    sys.path.insert(0, str(prob_dir))

    run_condition(args.problem, args.condition, args.seeds)
