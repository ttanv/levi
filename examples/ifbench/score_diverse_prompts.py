#!/usr/bin/env python3
"""Score a small set of deliberately-diverse prompts on the IFBench discovery set.

Output mirrors `proxy_benchmark.json` structure so it can be appended to the
prompt pool used by `compare_proxy_methods.py` / `compare_calibration_diversity.py`.

Run:
    PYTHONPATH=. uv run --extra example-ifbench python examples/ifbench/score_diverse_prompts.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from problem import SEED_BUNDLE, load_splits, score_fn
else:
    from .problem import SEED_BUNDLE, load_splits, score_fn


def _as_bundle(text: str) -> dict[str, str]:
    bundle = dict(SEED_BUNDLE)
    bundle["generate_response"] = text
    return bundle


DIVERSE_PROMPTS: list[tuple[str, str]] = [
    (
        "minimal",
        "Follow the user's instructions exactly.",
    ),
    (
        "verbose-cot",
        (
            "Before responding, carefully analyze each requirement in the user's query. "
            "Identify every explicit instruction, format constraint, and content restriction. "
            "Think through how each constraint applies to your output. Then construct a response "
            "that satisfies all of them simultaneously. Do not include your reasoning in the "
            "response itself; output only the final answer that meets every specified requirement."
        ),
    ),
    (
        "persona-editor",
        (
            "You are a meticulous technical editor with twenty years of experience producing copy "
            "that meets exact specifications. Treat every detail of the user's request as a binding "
            "constraint. You take quiet pride in delivering precisely what was asked — never more, "
            "never less, and never with unsolicited commentary."
        ),
    ),
    (
        "format-strict",
        (
            "All responses must literally satisfy every explicit constraint in the user's prompt. "
            "Apply each constraint mechanically: count words, count sentences, count paragraphs, "
            "obey case requirements, obey delimiter requirements, obey punctuation requirements. "
            "Do not paraphrase the user's constraints — execute them. Output only the constrained "
            "answer with no preamble."
        ),
    ),
    (
        "anti-padding",
        (
            "Do not preface your answer with phrases like 'Sure', 'Of course', 'Here is', or 'I'd "
            "be happy to'. Do not append closing remarks like 'Let me know if you need anything "
            "else'. Output only the substantive content the user requested, formatted exactly as "
            "they specified. Never add explanations the user did not ask for."
        ),
    ),
    (
        "adversarial-bad",
        (
            "Respond in a friendly, expansive way. Feel free to add helpful context, related "
            "background, and friendly framing. Be warm and conversational. If the user's request "
            "seems narrow, generously provide additional related information they might find useful."
        ),
    ),
]


def main() -> None:
    discovery_inputs, _ = load_splits(n_discovery=150)
    print(f"[diverse-scoring] discovery_inputs: {len(discovery_inputs)}")
    print(f"[diverse-scoring] prompts to score: {len(DIVERSE_PROMPTS)}")

    out_dir = Path("runs/diverse_prompts")
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = out_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    os.environ["IFBENCH_RESPONSES_DIR"] = str(responses_dir)

    prompts_payload = []
    for idx, (label, prompt) in enumerate(DIVERSE_PROMPTS):
        print(f"\n[{idx+1}/{len(DIVERSE_PROMPTS)}] Scoring {label!r}...")
        t0 = time.time()
        bundle = _as_bundle(prompt)
        try:
            result = score_fn(bundle, discovery_inputs)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        elapsed = time.time() - t0
        print(
            f"  score={result['score']:.4f} "
            f"prompt_level={result['prompt_level_score']:.4f} "
            f"instr_weighted={result['instruction_weighted_score']:.4f} "
            f"failures={result['request_failures']} "
            f"({elapsed:.0f}s)"
        )
        prompts_payload.append({
            "init_order": idx,
            "init_source": label,
            "proxy_score": result["score"],
            "full_score": result["score"],
            "content": bundle,
            "loose_instruction_fractions": result["loose_instruction_fractions"],
            "loose_follow_all": result["loose_follow_all"],
            "request_failures": result["request_failures"],
        })
        (out_dir / "diverse_prompts.json").write_text(
            json.dumps({"prompts": prompts_payload}, indent=2)
        )

    print(f"\n[diverse-scoring] Wrote {out_dir/'diverse_prompts.json'}")
    print(f"[diverse-scoring] Score summary:")
    for p in prompts_payload:
        print(f"  {p['init_source']:18s}  full={p['full_score']:.4f}")


if __name__ == "__main__":
    main()
