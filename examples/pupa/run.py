#!/usr/bin/env python3
"""Run Levi for PUPA / PAPILLON (gepa-artifact parity).

GEPA-comparable budget: 5 init seeds × 111 discovery = 555 rollouts plus
25 budget evals × 30 proxy = 750 rollouts ≈ ~1.3k rollouts (~half of GEPA's
2426-call PUPA budget for MIPROv2-Heavy).
"""

import json
from pathlib import Path

import levi
from problem import PROBLEM_DESCRIPTION, SEED_BUNDLE, load_splits, score_fn

RESUME_SNAPSHOT: Path | None = None


def main() -> None:
    trainset, testset = load_splits()
    resume = json.loads(RESUME_SNAPSHOT.read_text()) if RESUME_SNAPSHOT and RESUME_SNAPSHOT.exists() else None

    # Qwen3 thinking-mode sampling params (HuggingFace model card):
    # temperature=0.6, top_p=0.95, top_k=20, min_p=0.
    # max_tokens=8192 mirrors GEPA's `MAX_CONTEXT_LENGTH = 8192` cap
    # (gepa-artifact/scripts/experiment_configs.py) so qwen3-8b doesn't get
    # more thinking budget here than it has under the GEPA setup.
    proposer_lm = levi.LM(
        "openrouter/qwen/qwen3-8b",
        temperature=0.6,
        top_p=0.95,
        max_tokens=8192,
        extra_body={
            "provider": {"only": ["alibaba"]},
            "top_k": 20,
            "min_p": 0,
        },
    )

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
        budget_evals=25,
        resume_snapshot=resume,
        sampler_model_pairs=sampler_model_pairs,
        pipeline=levi.PipelineConfig(
            n_llm_workers=2,
            n_eval_processes=2,
            eval_timeout=7200.0,
        ),
        # PUPA has 2 components; bundle init runs n_diverse_seeds per component.
        # 1 (user seed) + 2 components × 2 seeds = 5 init candidates, each
        # scored on the full 111-example discovery set => 555 init rollouts.
        init=levi.InitConfig(n_diverse_seeds=2, n_variants_per_seed=0),
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
    print(f"Test score: {test_result['score']:.4f}  quality={test_result['quality_mean']:.4f}  leakage={test_result['leakage_mean']:.4f}")


if __name__ == "__main__":
    main()
