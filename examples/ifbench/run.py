#!/usr/bin/env python3
"""Evolve a system prompt on IFBench with Levi.

Example:
    PYTHONPATH=. uv run --with datasets --with git+https://github.com/allenai/IFBench.git \\
        python examples/ifbench/run.py --smoke-size 5 --smoke-only
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import levi

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from problem import PROBLEM_DESCRIPTION, SEED_BUNDLE, TASK_MODEL, load_splits, score_fn
else:
    from .problem import PROBLEM_DESCRIPTION, SEED_BUNDLE, TASK_MODEL, load_splits, score_fn


def _safe_score(label: str, bundle: dict[str, str], inputs: list[dict]) -> dict:
    try:
        result = score_fn(bundle, inputs)
    except Exception as error:
        print(f"[IFBench] {label} evaluation failed: {error}")
        return {"score": None, "error": str(error)}

    print(
        f"[IFBench] {label} score={result.get('score')} "
        f"prompt_level={result.get('prompt_level_score')} "
        f"instr_weighted={result.get('instruction_weighted_score')} "
        f"failures={result.get('request_failures')}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Levi prompt optimization on IFBench.")
    parser.add_argument("--n-discovery", type=int, default=150)
    parser.add_argument("--proxy-subset-size", type=int, default=40)
    parser.add_argument("--smoke-size", type=int, default=5)
    parser.add_argument("--budget-evals", type=int, default=30)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discovery_inputs, testset = load_splits(n_discovery=args.n_discovery)
    proxy_subset_size = min(args.proxy_subset_size, len(discovery_inputs))

    proposer_model = os.getenv(
        "IFBENCH_PROPOSER_MODEL",
        TASK_MODEL if TASK_MODEL.startswith("openrouter/") else f"openrouter/{TASK_MODEL}",
    )
    print(
        f"[IFBench] discovery={len(discovery_inputs)} test={len(testset)} "
        f"task_model={TASK_MODEL} proposer_model={proposer_model}"
    )

    smoke_inputs = discovery_inputs[: min(args.smoke_size, len(discovery_inputs))]
    if smoke_inputs:
        print(f"[IFBench] Smoke-checking seed bundle on {len(smoke_inputs)} discovery prompts...")
        _safe_score("Smoke seed", SEED_BUNDLE, smoke_inputs)
        baseline_bundle = {
            "generate_response": "Follow the user's instructions exactly. Do not add extra framing or formatting.",
            "ensure_correct_response": (
                "Re-read the user query and the draft response. Rewrite the response so every "
                "explicit constraint is satisfied; output only the corrected response."
            ),
        }
        _safe_score("Smoke baseline", baseline_bundle, smoke_inputs)

    if args.smoke_only:
        return

    output_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_ifbench"
    responses_dir = Path(output_dir) / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    os.environ["IFBENCH_RESPONSES_DIR"] = str(responses_dir)

    # Qwen3 official thinking-mode sampling params (HuggingFace model card):
    # temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=32768.
    proposer_lm = levi.LM(
        proposer_model,
        temperature=0.6,
        top_p=0.95,
        max_tokens=32768,
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
        inputs=discovery_inputs,
        model=proposer_lm,
        component_selector="ucb",
        budget_evals=args.budget_evals,
        sampler_model_pairs=sampler_model_pairs,
        pipeline=levi.PipelineConfig(
            n_llm_workers=2,
            n_eval_processes=2,
            eval_timeout=7200.0,
        ),
        init=levi.InitConfig(n_diverse_seeds=2, n_variants_per_seed=0),
        punctuated_equilibrium=levi.PunctuatedEquilibriumConfig(
            enabled=False, interval=20, component_selector="stagnation"
        ),
        proxy_benchmark=levi.ProxyBenchmarkConfig(
            enabled=True,
            discovery_inputs=discovery_inputs,
            matrix_key="per_example_scores",
            subset_size=proxy_subset_size,
        ),
        output_dir=output_dir,
    )

    best_bundle = result.best_bundle or dict(SEED_BUNDLE)
    print(
        f"[IFBench] Best proxy score: {result.best_score:.4f}  "
        f"({result.total_evaluations} evals, ${result.total_cost:.3f})"
    )
    print(f"[IFBench] Best bundle:\n{json.dumps(best_bundle, indent=2)}\n")

    print(f"[IFBench] Rescoring best bundle on full discovery set ({len(discovery_inputs)} problems)...")
    full_result = _safe_score("Full discovery rescore", best_bundle, discovery_inputs)
    full_score = full_result.get("score")
    proxy_full_gap = (full_score - result.best_score) if isinstance(full_score, (int, float)) else None
    if proxy_full_gap is not None:
        print(f"[IFBench] Proxy fidelity: proxy={result.best_score:.4f} full={full_score:.4f} gap={proxy_full_gap:+.4f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/test_result.json", "w") as f:
        json.dump(
            {
                "best_bundle": best_bundle,
                "proxy_score": result.best_score,
                "full_discovery_score": full_score,
                "full_discovery_result": full_result,
                "proxy_full_gap": proxy_full_gap,
                "total_evaluations": result.total_evaluations,
                "total_cost": result.total_cost,
            },
            f,
            indent=2,
        )
    print(f"[IFBench] Wrote {output_dir}/test_result.json")


if __name__ == "__main__":
    main()
