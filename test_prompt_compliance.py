#!/usr/bin/env python3
"""
Test LLM compliance with prompt instructions.
"""

import sys
import os
import re
from pathlib import Path

ALGOFORGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ALGOFORGE_ROOT))

import litellm
import logging

logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
litellm.suppress_debug_info = True
litellm.set_verbose = False

# Register models
litellm.register_model({
    "openrouter/openai/gpt-5.1-codex-mini": {
        "max_tokens": 32768,
        "max_input_tokens": 200000,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "openrouter",
    },
    "openrouter/google/gemini-2.5-flash-lite": {
        "max_tokens": 32768,
        "max_input_tokens": 1048576,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000004,
        "litellm_provider": "openrouter",
    },
    "openrouter/z-ai/glm-4.6": {
        "max_tokens": 32768,
        "max_input_tokens": 204800,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.00000039,
        "output_cost_per_token": 0.0000019,
        "litellm_provider": "openrouter",
    },
    "openrouter/x-ai/grok-code-fast-1": {
        "max_tokens": 32768,
        "max_input_tokens": 256000,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0000002,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "openrouter",
    },
    "openrouter/google/gemini-3-pro-preview": {
        "max_tokens": 32768,
        "max_input_tokens": 1048576,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000012,
        "litellm_provider": "openrouter",
    },
})

# Provider routing for specific models
PROVIDER_ROUTING = {
    "openrouter/z-ai/glm-4.6": {"provider": {"only": ["Cerebras"]}},
}

from algoforge.llm import PromptBuilder, ProgramWithScore, OutputMode
from algoforge.core import Program
from algoforge.methods.alphaevolve import extract_code

SAMPLE_CODE = '''import time
import random
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

def get_best_schedule(workload, num_seqs):
    seq = list(range(workload.num_txns))
    random.shuffle(seq)
    return workload.get_opt_seq_cost(seq), seq

def get_random_costs():
    random.seed(42)
    start = time.time()
    w1, w2, w3 = Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)
    c1, s1 = get_best_schedule(w1, 1)
    c2, s2 = get_best_schedule(w2, 1)
    c3, s3 = get_best_schedule(w3, 1)
    return c1 + c2 + c3, [s1, s2, s3], time.time() - start
'''

PROBLEM = """Optimize transaction scheduling for database workloads to minimize total makespan.
- Transactions have read (r) and write (w) operations on keys
- Write-Write and Read-Write conflicts require waiting
- Find the optimal ordering of 100 transactions to minimize makespan
"""

SIGNATURE = """def get_best_schedule(workload: Workload, num_seqs: int) -> tuple[int, list[int]]:
    '''Returns (makespan, schedule).'''

def get_random_costs() -> tuple[int, list[list[int]], float]:
    '''Returns (total_makespan, [sched1, sched2, sched3], time).'''
"""


def apply_diff_debug(original: str, diff_response: str) -> tuple[str | None, str]:
    """Apply SEARCH/REPLACE diff blocks with debug info."""
    result = original
    debug_lines = []

    pattern = r'<<<<<<< SEARCH\s*(.*?)\s*=======\s*(.*?)\s*>>>>>>> REPLACE'
    matches = re.findall(pattern, diff_response, re.DOTALL)

    if not matches:
        debug_lines.append("No SEARCH/REPLACE blocks found")
        extracted = extract_code(diff_response)
        if extracted:
            debug_lines.append(f"Fallback extract_code: {len(extracted)} chars")
            return extracted, "\n".join(debug_lines)
        return None, "\n".join(debug_lines)

    debug_lines.append(f"Found {len(matches)} blocks")

    for i, (search, replace) in enumerate(matches):
        search_stripped = search.strip()
        replace_stripped = replace.strip()

        if search_stripped in result:
            result = result.replace(search_stripped, replace_stripped, 1)
            debug_lines.append(f"Block {i+1}: OK")
        else:
            debug_lines.append(f"Block {i+1}: SEARCH not found")
            if 'def get_random_costs' in replace_stripped:
                return replace_stripped, "\n".join(debug_lines)
            return None, "\n".join(debug_lines)

    return result, "\n".join(debug_lines)


def test_model(model: str, n_trials: int = 3):
    """Test a single model - DIFF mode only."""
    print(f"\n{'='*70}")
    print(f"Testing: {model}")
    print(f"{'='*70}")

    results = {"diff": []}

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")

        # DIFF mode only
        print("[DIFF]", end=" ")
        builder = PromptBuilder()
        builder.add_section("Problem", PROBLEM, priority=10)
        builder.add_section("Signature", f"```python\n{SIGNATURE}\n```", priority=20)
        builder.add_parents([ProgramWithScore(Program(code=SAMPLE_CODE), None)], priority=30)
        builder.set_output_mode(OutputMode.DIFF)

        try:
            extra = PROVIDER_ROUTING.get(model, {})
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": builder.build()}],
                temperature=0.7,
                max_tokens=4096,
                extra_body=extra if extra else None,
            )
            content = response.choices[0].message.content
            result_code, debug = apply_diff_debug(SAMPLE_CODE, content)

            success = result_code is not None
            has_markers = "<<<<<<< SEARCH" in content and ">>>>>>> REPLACE" in content
            starts_ok = content.strip().startswith("<<<<<<< SEARCH") or content.strip().startswith("```")

            results["diff"].append({"success": success, "has_markers": has_markers, "starts_ok": starts_ok})
            print(f"{'OK' if success else 'FAIL'} | markers={has_markers} | starts_ok={starts_ok}")
            if not success:
                print(f"  Debug: {debug}")
                print(f"  Response ({len(content)} chars):")
                print(content)

        except Exception as e:
            results["diff"].append({"success": False, "error": str(e)})
            print(f"ERROR: {e}")

    diff_ok = sum(1 for r in results["diff"] if r.get("success", False))
    print(f"\n[{model.split('/')[-1]}] DIFF: {diff_ok}/{n_trials}")

    return results


def main():
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set")
        return

    MODELS = [
        "openrouter/openai/gpt-5.1-codex-mini",
        "openrouter/google/gemini-2.5-flash-lite",
        "openrouter/z-ai/glm-4.6",
        "openrouter/x-ai/grok-code-fast-1",
        "openrouter/google/gemini-3-pro-preview",
    ]

    print("=" * 70)
    print("LLM DIFF Mode Compliance Test (3 trials each)")
    print("=" * 70)

    all_results = {}
    for model in MODELS:
        all_results[model] = test_model(model, n_trials=3)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'DIFF':<10}")
    print("-" * 45)

    for model in MODELS:
        name = model.split("/")[-1][:29]
        r = all_results[model]
        diff_pct = sum(1 for x in r["diff"] if x.get("success")) / len(r["diff"]) * 100
        print(f"{name:<30} {diff_pct:>5.0f}%")


if __name__ == "__main__":
    main()
