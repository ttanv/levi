#!/usr/bin/env python3
"""
DSPy prompt optimization for EPLB mutations - v2.
Uses real elite programs and incorporates observed error patterns.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import dspy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from problem import (
    PROBLEM_DESCRIPTION,
    FUNCTION_SIGNATURE,
    get_lazy_inputs,
    score_fn,
)

# Model config
LIGHT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LIGHT_MODEL_ENDPOINT = "http://localhost:8000/v1"


def load_elite_programs(snapshot_path: str) -> list[tuple[str, float]]:
    """Load elite programs from a snapshot file."""
    with open(snapshot_path) as f:
        d = json.load(f)

    elites = []
    for e in d.get('elites', []):
        score = e.get('primary_score')
        code = e.get('code')
        if score is not None and code:
            elites.append((code, score))

    # Sort by score descending
    elites.sort(key=lambda x: x[1], reverse=True)
    return elites


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


# Enhanced problem description with common error hints
ENHANCED_PROBLEM = """# EPLB Load Balancer

Map 64 logical experts to 288 physical GPU slots for load balancing.

## Constraints (CRITICAL - violations = 0 score):
- expert_count.sum(dim=1) == 288 for ALL layers
- expert_count[layer, i] >= 1 for ALL experts (no zeros!)
- physical_to_logical_map: [layers, 288], values in [0, 63]
- logical_to_physical_map: [layers, 64, X], physical slot indices or -1
- ALL outputs must be dtype=torch.int64

## Common Errors to AVOID:
- Index 288 out of bounds (use indices 0-287)
- Duplicate physical slot assignments
- Wrong tensor shape/size
- Forgetting to ensure every expert has >=1 replica

## Scoring:
- 90% balancedness: avg_gpu_load / max_gpu_load
- 10% speed: faster is better
"""


# DSPy Signature with enhanced instructions
class MutationSignature(dspy.Signature):
    """Improve EPLB load balancing code using SEARCH/REPLACE diffs.

    Given a parent solution and inspiration, generate surgical code edits.
    The edits must preserve all EPLB constraints:
    - expert_count.sum() == 288 per layer
    - All experts need >= 1 replica
    - Indices: logical 0-63, physical 0-287
    - Output dtype: torch.int64
    """

    problem_description: str = dspy.InputField(desc="EPLB problem and constraints")
    parent_code: str = dspy.InputField(desc="Parent code to improve")
    parent_score: float = dspy.InputField(desc="Score of parent")
    inspiration_code: str = dspy.InputField(desc="Inspiration solution")
    inspiration_score: float = dspy.InputField(desc="Score of inspiration")

    diff_output: str = dspy.OutputField(
        desc="SEARCH/REPLACE blocks: <<<<<<< SEARCH\\nexact_lines\\n=======\\nreplacement\\n>>>>>>> REPLACE"
    )


class MutationModule(dspy.Module):
    """Module for generating code mutations."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(MutationSignature)

    def forward(self, problem_description: str, parent_code: str, parent_score: float,
                inspiration_code: str, inspiration_score: float) -> dspy.Prediction:
        return self.generate(
            problem_description=problem_description,
            parent_code=parent_code,
            parent_score=parent_score,
            inspiration_code=inspiration_code,
            inspiration_score=inspiration_score,
        )


def mutation_metric(example: dspy.Example, prediction: dspy.Prediction,
                    inputs: list, trace=None, debug: bool = False) -> float:
    """
    Evaluate a mutation with partial credit:
    - Has diff/code structure: 0.1
    - Applies correctly: +0.2
    - Compiles: +0.2
    - Runs without error: +0.2
    - Score maintained/improved: +0.3
    """
    parent_code = example.parent_code
    diff_output = prediction.diff_output
    score = 0.0

    # Check structure
    has_diff = "<<<<<<< SEARCH" in diff_output and "=======" in diff_output
    has_code = "```python" in diff_output or "```\n" in diff_output

    if has_diff or has_code:
        score += 0.1
        if debug:
            print(f"  [METRIC] Has structure: diff={has_diff}, code={has_code}")

    # Apply diff
    mutated_code = apply_diff(parent_code, diff_output)
    if mutated_code is None:
        if debug:
            print(f"  [METRIC] Diff failed to apply")
        return score

    score += 0.2
    if debug:
        print(f"  [METRIC] Diff applied")

    # Check compilation
    compiles, error = validate_code(mutated_code)
    if not compiles:
        if debug:
            print(f"  [METRIC] Doesn't compile: {error}")
        return score

    score += 0.2
    if debug:
        print(f"  [METRIC] Compiles")

    # Execute
    result = execute_code(mutated_code, inputs)
    if "error" in result:
        if debug:
            print(f"  [METRIC] Execution error: {result['error'][:50]}")
        return score

    score += 0.2
    child_score = result.get("score", 0.0)
    parent_score = example.parent_score

    if debug:
        print(f"  [METRIC] Runs. Parent: {parent_score:.2f}, Child: {child_score:.2f}")

    # Score improvement component
    if child_score > parent_score:
        improvement = (child_score - parent_score) / max(parent_score, 1.0)
        score += min(improvement * 2, 1.0) * 0.3
        if debug:
            print(f"  [METRIC] Improved!")
    elif child_score >= parent_score * 0.95:
        score += 0.15
        if debug:
            print(f"  [METRIC] Maintained")
    elif child_score >= parent_score * 0.8:
        score += 0.05

    return score


def generate_examples_from_elites(elites: list[tuple[str, float]], n_examples: int = 10) -> list[dspy.Example]:
    """Generate training examples from elite programs."""
    examples = []

    n_elites = len(elites)
    if n_elites < 2:
        raise ValueError("Need at least 2 elite programs")

    for i in range(n_examples):
        # Use different parent/inspiration pairs
        parent_idx = i % n_elites
        insp_idx = (i + 1) % n_elites

        parent_code, parent_score = elites[parent_idx]
        insp_code, insp_score = elites[insp_idx]

        example = dspy.Example(
            problem_description=ENHANCED_PROBLEM,
            parent_code=parent_code,
            parent_score=parent_score,
            inspiration_code=insp_code,
            inspiration_score=insp_score,
        ).with_inputs(
            "problem_description", "parent_code", "parent_score",
            "inspiration_code", "inspiration_score"
        )
        examples.append(example)

    return examples


def optimize_mutation_prompt(
    elites: list[tuple[str, float]],
    n_trials: int = 20,
    n_examples: int = 10,
    output_file: Optional[str] = None,
) -> dict:
    """Optimize mutation prompt using MIPROv2."""
    print("=" * 60)
    print("Optimizing MUTATION prompt with MIPROv2")
    print("=" * 60)

    # Setup
    lm = dspy.LM(
        model=f"openai/{LIGHT_MODEL}",
        api_base=LIGHT_MODEL_ENDPOINT,
        api_key="dummy",
        temperature=0.8,
        max_tokens=6000,
    )
    dspy.configure(lm=lm)

    inputs = get_lazy_inputs()

    def metric(example, prediction, trace=None):
        return mutation_metric(example, prediction, inputs, trace, debug=False)

    # Generate training examples
    trainset = generate_examples_from_elites(elites, n_examples)
    print(f"Generated {len(trainset)} training examples")

    # Create module
    module = MutationModule()

    # Setup MIPROv2
    from dspy.teleprompt import MIPROv2

    optimizer = MIPROv2(
        metric=metric,
        auto=None,
        num_candidates=5,
        verbose=True,
        num_threads=4,
    )

    print(f"\nRunning MIPROv2 optimization (zero-shot)...")

    optimized_module = optimizer.compile(
        module,
        trainset=trainset,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        num_trials=n_trials,
        minibatch=False,
        fewshot_aware_proposer=False,
    )

    # Extract optimized instructions
    optimized_instructions = {}
    for name, predictor in optimized_module.named_predictors():
        if hasattr(predictor, "signature"):
            optimized_instructions[name] = str(predictor.signature.instructions)

    # Evaluate
    testset = generate_examples_from_elites(elites, 5)
    scores = []
    for example in testset:
        try:
            pred = optimized_module(**example.inputs())
            score = mutation_metric(example, pred, inputs, debug=True)
            scores.append(score)
        except Exception as e:
            print(f"  Evaluation error: {e}")
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    result = {
        "target": "mutation_v2",
        "optimized_instructions": optimized_instructions,
        "avg_score": avg_score,
        "n_trials": n_trials,
        "n_examples": n_examples,
        "n_elites": len(elites),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\nOptimization complete!")
    print(f"Average test score: {avg_score:.3f}")
    print(f"\nOptimized instructions:")
    for name, instructions in optimized_instructions.items():
        print(f"\n{name}:\n{instructions}")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Optimize EPLB mutation prompts with DSPy")
    parser.add_argument("--snapshot", type=str, default="runs/20260122_072947/snapshot.json",
                       help="Path to snapshot with elite programs")
    parser.add_argument("--n-trials", type=int, default=15, help="Optimization trials")
    parser.add_argument("--n-examples", type=int, default=8, help="Training examples")
    parser.add_argument("--output-dir", type=str, default="prompt_optimization_results",
                       help="Output directory")
    args = parser.parse_args()

    # Find snapshot
    base = Path(__file__).parent.parent.parent
    # Also try relative to cwd
    if Path(args.snapshot).exists():
        snapshot_path = Path(args.snapshot)
    else:
        snapshot_path = base / args.snapshot
    if not snapshot_path.exists():
        print(f"Snapshot not found: {snapshot_path}")
        sys.exit(1)

    # Load elites
    print(f"Loading elites from: {snapshot_path}")
    elites = load_elite_programs(str(snapshot_path))
    print(f"Loaded {len(elites)} elite programs")
    print(f"Score range: {elites[-1][1]:.2f} - {elites[0][1]:.2f}")

    # Filter to top elites (positive scores only)
    elites = [(c, s) for c, s in elites if s > 0]
    print(f"Using {len(elites)} elites with positive scores")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"mutation_v2_{timestamp}.json"

    # Run optimization
    optimize_mutation_prompt(
        elites=elites,
        n_trials=args.n_trials,
        n_examples=args.n_examples,
        output_file=str(output_file),
    )


if __name__ == "__main__":
    main()
