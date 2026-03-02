#!/usr/bin/env python3
"""
Feature Selection for TXN Scheduling
=====================================

1. Run init phase (diverse seeds + variants) to generate a pool of programs
2. Extract ALL 13 AST features from each program
3. PCA + correlation analysis to find the 4 most informative, uncorrelated features
4. Print the recommended feature set
"""

import sys
import asyncio
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

import numpy as np

# Add txn_scheduling to path
sys.path.insert(0, str(Path(__file__).parent.parent / "txn_scheduling"))

from algoforge import (
    AlgoforgeConfig, BudgetConfig,
    InitConfig, MetaAdviceConfig, PipelineConfig, CVTConfig, BehaviorConfig,
    PunctuatedEquilibriumConfig, PromptOptConfig,
)
from algoforge.behavior.extractor import BehaviorExtractor
from algoforge.core import Program
from algoforge.init.diversifier import Diversifier
from algoforge.utils import ResilientProcessPool, evaluate_code, extract_fn_name
from algoforge.pipeline.state import PipelineState
from algoforge.llm import set_llm_client, clear_llm_client, UnifiedLLMClient
from algoforge.llm.unified_client import UnifiedLLMClientConfig
from algoforge.methods.algoforge import _register_models_with_litellm

import problem

import ast as ast_module

ALL_FEATURES = [
    "code_length", "ast_depth", "cyclomatic_complexity", "loop_count",
    "math_operators", "branch_count", "loop_nesting_max", "function_def_count",
    "numeric_literal_count", "comparison_count", "subscript_count",
    "call_count", "comprehension_count", "range_max_arg",
]

FEATURE_FUNCS = BehaviorExtractor.BUILT_IN_FEATURES

QWEN_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def extract_all_features(code: str) -> dict[str, float]:
    """Extract all 13+1 AST features from a program."""
    try:
        tree = ast_module.parse(code)
    except SyntaxError:
        return {}
    prog = Program(content=code)
    features = {}
    for name in ALL_FEATURES:
        fn = FEATURE_FUNCS.get(name)
        if fn:
            try:
                if name == "code_length":
                    features[name] = fn(prog, None)
                else:
                    features[name] = fn(prog, tree)
            except Exception:
                features[name] = 0.0
    return features


async def run_init_phases():
    """Run init phases 1 & 2, return list of (code, score, result) tuples."""
    fn_name = extract_fn_name(problem.FUNCTION_SIGNATURE)

    config = AlgoforgeConfig(
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
        budget=BudgetConfig(evaluations=1000),
        cvt=CVTConfig(n_centroids=50, defer_centroids=True),
        init=InitConfig(
            enabled=True,
            n_diverse_seeds=5,
            n_variants_per_seed=8,
            temperature=0.8,
            diversity_prompt=problem.DIVERSITY_SEED_PROMPT,
        ),
        meta_advice=MetaAdviceConfig(enabled=False),
        pipeline=PipelineConfig(
            n_llm_workers=4,
            n_eval_processes=4,
            eval_timeout=180.0,
            max_tokens=8192,
        ),
        behavior=BehaviorConfig(ast_features=ALL_FEATURES, score_keys=[], init_noise=0.0),
        punctuated_equilibrium=PunctuatedEquilibriumConfig(enabled=False),
        prompt_opt=PromptOptConfig(enabled=False),
        output_dir="runs/ablations/feature_selection",
    )

    _register_models_with_litellm(config)
    llm_client = UnifiedLLMClient(UnifiedLLMClientConfig(max_tokens=8192))
    set_llm_client(llm_client)

    extractor = BehaviorExtractor(ast_features=ALL_FEATURES, score_keys=[], init_noise=0.0)
    executor = ResilientProcessPool(max_workers=4)
    state = PipelineState(config.budget)

    try:
        # Evaluate seed
        seed_result = await executor.run(
            evaluate_code, config.seed_program, config.score_fn,
            config.inputs, fn_name, timeout=180.0,
        )
        seed_score = seed_result.get('score', 0.0)
        print(f"Seed score: {seed_score:.1f}")
        state.record_accept()

        # Run diversifier phases 1 & 2 only
        diversifier = Diversifier(config, executor, state=state)
        diverse_seeds = await diversifier._generate_diverse_seeds(
            config.seed_program, seed_score, seed_result, fn_name
        )
        print(f"\nPhase 1: {len(diverse_seeds)} diverse seeds")

        valid_programs, behavior_vectors = await diversifier._generate_variants(
            diverse_seeds, fn_name, extractor
        )
        print(f"Phase 2: {len(valid_programs)} valid programs")

        return valid_programs

    finally:
        await llm_client.close()
        clear_llm_client()
        executor.shutdown()


def analyze_features(valid_programs: list[dict]):
    """Analyze all features across programs to find the best 4."""
    print(f"\n{'='*70}")
    print(f"Feature Analysis on {len(valid_programs)} programs")
    print(f"{'='*70}")

    # Extract all features from each program
    all_features = []
    for prog in valid_programs:
        feats = extract_all_features(prog["code"])
        if feats:
            all_features.append(feats)

    print(f"Successfully extracted features from {len(all_features)} programs\n")

    if len(all_features) < 5:
        print("ERROR: Too few programs for meaningful analysis")
        return

    # Build matrix: rows = programs, cols = features
    feature_names = ALL_FEATURES
    matrix = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features])

    # --- Step 1: Basic stats ---
    print("Feature Statistics:")
    print(f"{'Feature':30s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Unique':>8s}")
    print("-" * 80)
    for i, name in enumerate(feature_names):
        col = matrix[:, i]
        n_unique = len(np.unique(col))
        print(f"{name:30s} {col.mean():8.1f} {col.std():8.2f} {col.min():8.1f} {col.max():8.1f} {n_unique:8d}")

    # --- Step 2: Remove zero-variance features ---
    stds = matrix.std(axis=0)
    valid_mask = stds > 1e-6
    valid_names = [name for name, v in zip(feature_names, valid_mask) if v]
    valid_matrix = matrix[:, valid_mask]
    valid_stds = stds[valid_mask]

    print(f"\nFeatures with variance: {len(valid_names)}")
    if len(valid_names) < 4:
        print("ERROR: fewer than 4 features have variance")
        return

    # --- Step 3: Z-score normalize ---
    means = valid_matrix.mean(axis=0)
    normed = (valid_matrix - means) / valid_stds

    # --- Step 4: Correlation matrix ---
    corr = np.corrcoef(normed.T)
    print(f"\nCorrelation Matrix (z-scored):")
    print(f"{'':30s}", end="")
    for name in valid_names:
        print(f" {name[:8]:>8s}", end="")
    print()
    for i, name in enumerate(valid_names):
        print(f"{name:30s}", end="")
        for j in range(len(valid_names)):
            print(f" {corr[i,j]:8.2f}", end="")
        print()

    # --- Step 5: PCA ---
    from numpy.linalg import eigh
    cov = np.cov(normed.T)
    eigenvalues, eigenvectors = eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"\nPCA - Explained Variance:")
    total_var = eigenvalues.sum()
    cumulative = 0.0
    for i, ev in enumerate(eigenvalues):
        cumulative += ev
        pct = ev / total_var * 100
        cum_pct = cumulative / total_var * 100
        print(f"  PC{i+1}: {pct:5.1f}% (cumulative: {cum_pct:5.1f}%)")
        # Show top loadings
        loadings = eigenvectors[:, i]
        top_idx = np.argsort(np.abs(loadings))[::-1][:3]
        for ti in top_idx:
            print(f"    {valid_names[ti]:30s} loading={loadings[ti]:+.3f}")

    # --- Step 6: Greedy selection of 4 most diverse features ---
    # Strategy: pick feature with highest variance, then iteratively pick
    # the feature with highest variance that has lowest max-correlation with selected
    print(f"\n{'='*70}")
    print("Greedy Feature Selection (high variance, low correlation)")
    print(f"{'='*70}")

    # Coefficient of variation (relative variance) - better for comparing across scales
    cv = valid_stds / (np.abs(means) + 1e-6)

    selected = []
    remaining = list(range(len(valid_names)))

    # First: pick highest-variance feature
    best_idx = max(remaining, key=lambda i: cv[i])
    selected.append(best_idx)
    remaining.remove(best_idx)
    print(f"\n  1. {valid_names[best_idx]:30s} (CV={cv[best_idx]:.2f})")

    # Subsequent: pick highest variance with low correlation to all selected
    for pick in range(2, 5):
        best_score = -1
        best_j = -1
        for j in remaining:
            max_corr = max(abs(corr[j, s]) for s in selected)
            # Score = variance * (1 - max_correlation_with_selected)
            score = cv[j] * (1 - max_corr)
            if score > best_score:
                best_score = score
                best_j = j
        selected.append(best_j)
        remaining.remove(best_j)
        max_corr = max(abs(corr[best_j, s]) for s in selected if s != best_j)
        print(f"  {pick}. {valid_names[best_j]:30s} (CV={cv[best_j]:.2f}, max_corr={max_corr:.2f})")

    selected_names = [valid_names[i] for i in selected]
    print(f"\n  RECOMMENDED FEATURES: {selected_names}")

    # Save result
    result = {
        "selected_features": selected_names,
        "all_feature_stats": {
            name: {
                "mean": float(matrix[:, i].mean()),
                "std": float(matrix[:, i].std()),
                "cv": float(cv[list(valid_names).index(name)]) if name in valid_names else 0.0,
            }
            for i, name in enumerate(feature_names)
        },
        "n_programs": len(all_features),
    }
    out_path = Path("runs/ablations/feature_selection/analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Saved to {out_path}")

    return selected_names


def main():
    print("Running init phase to generate programs for feature analysis...")
    valid_programs = asyncio.run(run_init_phases())

    if not valid_programs:
        print("ERROR: No valid programs generated")
        sys.exit(1)

    selected = analyze_features(valid_programs)
    print(f"\n\nFinal recommendation: {selected}")


if __name__ == "__main__":
    main()
