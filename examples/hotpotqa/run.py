#!/usr/bin/env python3
"""Run Levi for HotPotQA (gepa-artifact parity)."""

import json
from pathlib import Path

import levi
from problem import PROBLEM_DESCRIPTION, SEED_BUNDLE, load_splits, score_fn

RESUME_SNAPSHOT: Path | None = None


def main() -> None:
    trainset, testset = load_splits()
    resume = json.loads(RESUME_SNAPSHOT.read_text()) if RESUME_SNAPSHOT and RESUME_SNAPSHOT.exists() else None

    # Qwen3 official thinking-mode sampling params (HuggingFace model card):
    # temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=32768.
    proposer_lm = levi.LM(
        "openrouter/qwen/qwen3-8b",
        temperature=0.6,
        top_p=0.95,
        max_tokens=32768,
        extra_body={
            "provider": {"only": ["alibaba"]},
            "top_k": 20,
            "min_p": 0,
        },
    )

    # Sampler ablation: drop T=1.0 and T=1.2 (0/14 NEW BEST across two trajectories).
    # Keep T=0.3 (workhorse: 4/15 NEW BEST, max 0.80) and T=0.7 (seeder: 2/8 NEW BEST).
    sampler_model_pairs = [
        levi.SamplerModelPair(sampler="softmax", model=proposer_lm, weight=1.0, temperature=0.3),
        levi.SamplerModelPair(sampler="softmax", model=proposer_lm, weight=1.0, temperature=0.7),
    ]

    result = levi.evolve_prompts(
        PROBLEM_DESCRIPTION,
        evaluator=score_fn,
        seed_prompt=SEED_BUNDLE,
        inputs=trainset,
        model=proposer_lm,
        component_selector="ucb",
        budget_evals=60,
        resume_snapshot=resume,
        sampler_model_pairs=sampler_model_pairs,
        pipeline=levi.PipelineConfig(
            n_llm_workers=2,
            n_eval_processes=2,
            eval_timeout=7200.0,
        ),
        init=levi.InitConfig(n_diverse_seeds=1, n_variants_per_seed=0),
        punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(
            enabled=False, interval=20, component_selector="stagnation"
        ),
        proxy_benchmark=levi.ProxyBenchmarkConfig(
            enabled=True,
            discovery_inputs=trainset,
            matrix_key="per_example_scores",
            subset_size=40,
        ),
    )

    print(f"Best proxy score: {result.best_score:.4f}  ({result.total_evaluations} evals, ${result.total_cost:.3f})")

    best_bundle = result.best_bundle or {"prompt": result.best_prompt}
    test_result = score_fn(best_bundle, testset)
    print(f"Test EM: {test_result['em']:.4f}")


if __name__ == "__main__":
    main()
