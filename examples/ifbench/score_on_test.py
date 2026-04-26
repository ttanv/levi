#!/usr/bin/env python3
"""Score the best evolved prompt + best init prompt on the IFBench test set (300 held-out).

Usage:
    PYTHONPATH=. uv run --extra example-ifbench python examples/ifbench/score_on_test.py [run_dir]

Defaults to the latest runs/*_ifbench dir if not given.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from problem import BUNDLE_KEYS, SEED_BUNDLE, load_splits, score_fn
else:
    from .problem import BUNDLE_KEYS, SEED_BUNDLE, load_splits, score_fn


def _coerce_bundle(content: str | dict) -> dict[str, str]:
    """Accept legacy string content or a bundle dict/JSON payload."""
    if isinstance(content, dict):
        bundle = dict(SEED_BUNDLE)
        bundle.update({k: v for k, v in content.items() if k in BUNDLE_KEYS})
        return bundle
    if not isinstance(content, str):
        raise TypeError(f"Unexpected content type: {type(content).__name__}")

    text = content.strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            prompts = payload.get("prompts") if "prompts" in payload else payload
            if isinstance(prompts, dict):
                return _coerce_bundle(prompts)

    bundle = dict(SEED_BUNDLE)
    bundle["generate_response"] = content
    return bundle


def _default_run_dir() -> Path:
    candidates = sorted(Path("runs").glob("*_ifbench"))
    if not candidates:
        raise SystemExit("No runs/*_ifbench directories found.")
    return candidates[-1]


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_run_dir()
    print(f"[Test] Using run dir: {run_dir}")

    with open(run_dir / "proxy_benchmark.json") as f:
        pb = json.load(f)
    with open(run_dir / "test_result.json") as f:
        tr = json.load(f)

    best_init = max(pb["prompts"], key=lambda p: p["full_score"])
    evolved = tr.get("best_bundle") or tr.get("best_prompt")
    candidates = [
        (
            f"best init (init_order={best_init['init_order']}, full={best_init['full_score']:.4f})",
            _coerce_bundle(best_init["content"]),
        ),
        (
            f"evolved best (proxy={tr['proxy_score']:.4f}, full={tr['full_discovery_score']:.4f})",
            _coerce_bundle(evolved),
        ),
    ]

    _, testset = load_splits(n_discovery=150)
    print(f"[Test] testset size: {len(testset)}")

    # Persist responses for these test-set evals to a sibling subdir.
    os.environ["IFBENCH_RESPONSES_DIR"] = str(run_dir / "test_responses")

    results = {}
    for label, bundle in candidates:
        print(f"\n[Test] Scoring {label!r} on {len(testset)} test problems...")
        result = score_fn(bundle, testset)
        print(
            f"[Test] {label}: score={result['score']:.4f} "
            f"prompt_level={result['prompt_level_score']:.4f} "
            f"instr_weighted={result['instruction_weighted_score']:.4f} "
            f"failures={result['request_failures']}"
        )
        results[label] = {
            "bundle": bundle,
            "score": result["score"],
            "prompt_level_score": result["prompt_level_score"],
            "instruction_weighted_score": result["instruction_weighted_score"],
            "loose_instruction_fractions": result["loose_instruction_fractions"],
            "loose_follow_all": result["loose_follow_all"],
            "request_failures": result["request_failures"],
        }

    out_path = run_dir / "test_set_scores.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Test] Wrote {out_path}")


if __name__ == "__main__":
    main()
