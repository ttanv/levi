#!/usr/bin/env python3
"""
Analyze mutation quality from QWEN 30B.
Test different prompt formulations and collect error patterns.
"""

import sys
import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from openai import OpenAI

from problem import (
    PROBLEM_DESCRIPTION,
    FUNCTION_SIGNATURE,
    SEED_PROGRAM,
    get_lazy_inputs,
    score_fn,
)

# Model config
LIGHT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ENDPOINT = "http://localhost:8000/v1"


@dataclass
class MutationResult:
    prompt_name: str
    raw_output: str
    has_diff_structure: bool = False
    diff_applies: bool = False
    compiles: bool = False
    runs: bool = False
    score: Optional[float] = None
    error: Optional[str] = None
    error_category: Optional[str] = None


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from a response."""
    pattern = r'```python\s*(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    pattern = r'```\s*(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        code = matches[0].strip()
        if not code.startswith(('json', 'bash', 'shell')):
            return code
    return None


def apply_diff(original: str, diff_response: str) -> Optional[str]:
    """Apply SEARCH/REPLACE diff blocks to original code."""
    result = original

    pattern = r'<<<<<<< SEARCH\s*(.*?)\s*=======\s*(.*?)\s*>>>>>>> REPLACE'
    matches = re.findall(pattern, diff_response, re.DOTALL)

    if not matches:
        return extract_code(diff_response)

    applied_any = False
    for search, replace in matches:
        search = search.strip()
        replace = replace.strip()
        if search in result:
            result = result.replace(search, replace, 1)
            applied_any = True

    if not applied_any:
        return extract_code(diff_response)

    return result


def validate_code(code: str) -> tuple[bool, Optional[str]]:
    """Check if code compiles."""
    try:
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"Syntax: {e.msg}"
    except Exception as e:
        return False, str(e)


def execute_code(code: str, inputs: list) -> dict:
    """Execute code and return score_fn results."""
    namespace = {}
    try:
        exec(code, namespace)
        if "rebalance_experts" not in namespace:
            return {"error": "Function 'rebalance_experts' not found"}
        return score_fn(namespace["rebalance_experts"], inputs)
    except Exception as e:
        return {"error": str(e)}


def categorize_error(error: str) -> str:
    """Categorize error into high-level types."""
    error_lower = error.lower()
    if "syntax" in error_lower:
        return "syntax_error"
    elif "shape" in error_lower or "phy2log" in error_lower:
        return "wrong_shape"
    elif "logcnt" in error_lower and "288" in error_lower:
        return "logcnt_sum_wrong"
    elif "unhandled" in error_lower:
        return "unhandled_load"
    elif "negative" in error_lower:
        return "negative_values"
    elif "not found" in error_lower:
        return "function_not_found"
    elif "index" in error_lower or "out of" in error_lower:
        return "index_error"
    elif "attribute" in error_lower:
        return "attribute_error"
    elif "type" in error_lower:
        return "type_error"
    else:
        return "other"


INSPIRATION_1 = '''"""Greedy load-proportional replication."""
import torch

def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_layers, num_logical = weight.shape
    device = weight.device

    # Each expert gets at least 1 replica
    expert_count = torch.ones(num_layers, num_logical, dtype=torch.int64, device=device)
    remaining = num_replicas - num_logical

    # Distribute extra replicas to highest-load experts
    for _ in range(remaining):
        load_per_replica = weight / expert_count.float().clamp(min=1)
        max_idx = load_per_replica.argmax(dim=1)
        for layer in range(num_layers):
            expert_count[layer, max_idx[layer]] += 1

    # Build physical_to_logical_map
    phy2log = torch.zeros(num_layers, num_replicas, dtype=torch.int64, device=device)
    log2phy_list = []

    for layer in range(num_layers):
        idx = 0
        layer_log2phy = []
        for expert in range(num_logical):
            count = expert_count[layer, expert].item()
            slots = list(range(idx, idx + count))
            phy2log[layer, idx:idx+count] = expert
            layer_log2phy.append(slots)
            idx += count
        log2phy_list.append(layer_log2phy)

    max_replicas = expert_count.max().item()
    log2phy = torch.full((num_layers, num_logical, max_replicas), -1, dtype=torch.int64, device=device)
    for layer in range(num_layers):
        for expert in range(num_logical):
            slots = log2phy_list[layer][expert]
            log2phy[layer, expert, :len(slots)] = torch.tensor(slots, dtype=torch.int64)

    return phy2log, log2phy, expert_count
'''


# Different prompt variants to test
PROMPTS = {
    "baseline_diff": """## Problem
{problem}

## Function Signature
```python
{signature}
```

## v1 (Parent - Score: {parent_score:.1f})
```python
{parent_code}
```

## v2 (Inspiration - Score: {insp_score:.1f})
```python
{insp_code}
```

## Output
Output your improved code using SEARCH/REPLACE blocks.

FORMAT:
<<<<<<< SEARCH
exact lines to find
=======
replacement lines
>>>>>>> REPLACE

RULES:
1. Make SURGICAL changes - small, focused edits
2. Copy the SEARCH section EXACTLY from the original
3. Start immediately with <<<<<<< SEARCH
4. Do NOT include any explanation
""",

    "eplb_aware_diff": """## Expert Load Balancing (EPLB) Mutation

### Constraints (CRITICAL - violations = 0 score):
- `expert_count.sum(dim=1)` MUST equal 288 for ALL layers
- Every logical expert (0-63) MUST have at least 1 replica
- `physical_to_logical_map` must have shape [layers, 288] with values 0-63
- `logical_to_physical_map` must have shape [layers, 64, X] with slot indices or -1
- ALL output tensors must be torch.int64

### Problem
{problem}

### Function Signature
```python
{signature}
```

### v1 (Parent - Score: {parent_score:.1f})
```python
{parent_code}
```

### v2 (Inspiration - Score: {insp_score:.1f})
```python
{insp_code}
```

### Output
Improve v1's score by making targeted edits. Use SEARCH/REPLACE blocks:

<<<<<<< SEARCH
exact lines from v1
=======
replacement lines
>>>>>>> REPLACE

Start immediately with <<<<<<< SEARCH. No explanations.
""",

    "simple_full_code": """## Expert Load Balancing

### Critical Constraints:
- expert_count.sum(dim=1) == 288 for ALL layers
- Every expert needs at least 1 replica (no zeros in expert_count)
- Output tensors: torch.int64

### Problem
{problem}

### Function Signature
```python
{signature}
```

### Reference Solutions
v1 (Score: {parent_score:.1f}):
```python
{parent_code}
```

v2 (Score: {insp_score:.1f}):
```python
{insp_code}
```

### Task
Write an improved `rebalance_experts` function that achieves better load balancing.
Output ONLY the complete Python code in a ```python block. No explanations.
""",

    "error_focused": """## EPLB Code Improvement

### Common Errors to AVOID:
1. expert_count sums != 288 (most common failure!)
2. Some experts getting 0 replicas
3. Wrong tensor dtypes (must be torch.int64)
4. Index out of bounds

### Constraints:
- Input: weight [num_layers, 64]
- Output expert_count: [num_layers, 64], sum per layer = 288
- Output phy2log: [num_layers, 288], values 0-63
- Output log2phy: [num_layers, 64, X], physical slots or -1

### Problem
{problem}

### v1 (Parent - Score: {parent_score:.1f})
```python
{parent_code}
```

### v2 (Inspiration - Score: {insp_score:.1f})
```python
{insp_code}
```

### Output (SEARCH/REPLACE blocks)
Make targeted improvements to v1. Start with <<<<<<< SEARCH
""",

    "concise_constraints": """Improve this EPLB code. Constraints: expert_count.sum(dim=1)==288 per layer, no zeros, all int64.

Parent (Score: {parent_score:.1f}):
```python
{parent_code}
```

Inspiration (Score: {insp_score:.1f}):
```python
{insp_code}
```

Output SEARCH/REPLACE blocks only:
<<<<<<< SEARCH
exact lines
=======
replacement
>>>>>>> REPLACE
""",
}


def run_mutation(client: OpenAI, prompt_name: str, prompt: str, parent_code: str, inputs: list) -> MutationResult:
    """Run a single mutation and analyze result."""
    try:
        response = client.chat.completions.create(
            model=LIGHT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=4096,
        )
        raw_output = response.choices[0].message.content
    except Exception as e:
        return MutationResult(prompt_name=prompt_name, raw_output="", error=str(e), error_category="api_error")

    result = MutationResult(prompt_name=prompt_name, raw_output=raw_output)

    # Check for diff structure
    result.has_diff_structure = "<<<<<<< SEARCH" in raw_output and "=======" in raw_output

    # Try to apply diff
    mutated = apply_diff(parent_code, raw_output)
    if mutated is None:
        result.error = "Diff failed to apply"
        result.error_category = "diff_failed"
        return result
    result.diff_applies = True

    # Check compilation
    compiles, comp_error = validate_code(mutated)
    if not compiles:
        result.error = comp_error
        result.error_category = "syntax_error"
        return result
    result.compiles = True

    # Execute
    exec_result = execute_code(mutated, inputs)
    if "error" in exec_result:
        result.error = exec_result["error"]
        result.error_category = categorize_error(exec_result["error"])
        return result

    result.runs = True
    result.score = exec_result.get("score", 0.0)
    return result


def main():
    client = OpenAI(base_url=ENDPOINT, api_key="dummy")
    inputs = get_lazy_inputs()

    # Get parent score
    parent_result = execute_code(SEED_PROGRAM, inputs)
    parent_score = parent_result.get("score", 0.0) if "error" not in parent_result else 50.0

    insp_result = execute_code(INSPIRATION_1, inputs)
    insp_score = insp_result.get("score", 0.0) if "error" not in insp_result else 40.0

    print(f"Parent (SEED) score: {parent_score:.2f}")
    print(f"Inspiration score: {insp_score:.2f}")
    print()

    # Run experiments
    n_trials = 3
    all_results = []

    for prompt_name, prompt_template in PROMPTS.items():
        print(f"Testing prompt: {prompt_name}")
        prompt = prompt_template.format(
            problem=PROBLEM_DESCRIPTION,
            signature=FUNCTION_SIGNATURE,
            parent_code=SEED_PROGRAM,
            parent_score=parent_score,
            insp_code=INSPIRATION_1,
            insp_score=insp_score,
        )

        # Check prompt length
        print(f"  Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")

        for trial in range(n_trials):
            result = run_mutation(client, prompt_name, prompt, SEED_PROGRAM, inputs)
            all_results.append(result)

            status = "✓" if result.runs else "✗"
            if result.runs:
                print(f"  Trial {trial+1}: {status} score={result.score:.2f}")
            else:
                print(f"  Trial {trial+1}: {status} {result.error_category}: {result.error[:50] if result.error else 'N/A'}...")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for prompt_name in PROMPTS.keys():
        prompt_results = [r for r in all_results if r.prompt_name == prompt_name]
        n_total = len(prompt_results)
        n_runs = sum(1 for r in prompt_results if r.runs)
        n_diff = sum(1 for r in prompt_results if r.has_diff_structure)
        n_applies = sum(1 for r in prompt_results if r.diff_applies)
        n_compiles = sum(1 for r in prompt_results if r.compiles)
        avg_score = sum(r.score for r in prompt_results if r.score is not None) / max(n_runs, 1)

        print(f"\n{prompt_name}:")
        print(f"  Diff structure: {n_diff}/{n_total}")
        print(f"  Diff applies: {n_applies}/{n_total}")
        print(f"  Compiles: {n_compiles}/{n_total}")
        print(f"  Runs: {n_runs}/{n_total}")
        print(f"  Avg score: {avg_score:.2f}")

        errors = [r.error_category for r in prompt_results if r.error_category]
        if errors:
            print(f"  Errors: {Counter(errors)}")


if __name__ == "__main__":
    main()
