#!/usr/bin/env python3
"""
DSPy prompt optimization for AlgoForge.

Optimizes two prompts using COPRO:
1. Mutation generation prompt (light models) - for generating code diffs
2. Paradigm shift prompt (heavy model) - for generating novel approaches

Usage:
    python optimize_prompts.py --target mutation --n-trials 20
    python optimize_prompts.py --target paradigm --n-trials 10
    python optimize_prompts.py --target both --n-trials 20
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import dspy
import numpy as np
import torch

# Add algoforge to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from problem import (
    PROBLEM_DESCRIPTION,
    FUNCTION_SIGNATURE,
    SEED_PROGRAM,
    get_lazy_inputs,
    score_fn,
)

# --- Model Configuration (same as run_algoforge.py) ---
LIGHT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LIGHT_MODEL_ENDPOINT = "http://localhost:8000/v1"

HEAVY_MODEL = "deepseek/deepseek-v3.2"
HEAVY_MODEL_ENDPOINT = "https://openrouter.ai/api/v1"

# OpenRouter API key from environment
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


# --- Code Utilities ---
def extract_code(response: str) -> Optional[str]:
    """Extract Python code from a response."""
    # Try to find ```python blocks first
    pattern = r'```python\s*(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try generic code blocks
    pattern = r'```\s*(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        code = matches[0].strip()
        # Skip if it looks like a non-Python block
        if code.startswith(('json', 'bash', 'shell')):
            return None
        return code

    return None


def apply_diff(original: str, diff_response: str) -> Optional[str]:
    """Apply SEARCH/REPLACE diff blocks to original code."""
    result = original

    pattern = r'<<<<<<< SEARCH\s*(.*?)\s*=======\s*(.*?)\s*>>>>>>> REPLACE'
    matches = re.findall(pattern, diff_response, re.DOTALL)

    if not matches:
        # No diff blocks found, try to extract full code
        return extract_code(diff_response)

    applied_any = False
    for search, replace in matches:
        search = search.strip()
        replace = replace.strip()
        if search in result:
            result = result.replace(search, replace, 1)
            applied_any = True
        else:
            # Try with normalized whitespace (common issue with LLM output)
            search_normalized = ' '.join(search.split())
            for line in result.split('\n'):
                line_normalized = ' '.join(line.strip().split())
                if search_normalized in line_normalized or line_normalized in search_normalized:
                    # Found approximate match - try line-by-line matching
                    break
            # Search block not found - try to continue with other blocks
            continue

    # If no diffs applied successfully, try extracting full code
    if not applied_any:
        return extract_code(diff_response)

    return result


def validate_code(code: str) -> tuple[bool, Optional[str]]:
    """Check if code compiles without errors."""
    try:
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
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


def compute_code_features(code: str) -> dict[str, float]:
    """Extract behavioral features from code for diversity measurement."""
    import ast

    features = {
        "loop_count": 0,
        "branch_count": 0,
        "math_ops": 0,
        "call_count": 0,
        "comprehension_count": 0,
        "nesting_depth": 0,
    }

    try:
        tree = ast.parse(code)
    except:
        return features

    def count_nesting(node, depth=0):
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While, ast.If, ast.With)):
                child_depth = count_nesting(child, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = count_nesting(child, depth)
                max_depth = max(max_depth, child_depth)
        return max_depth

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            features["loop_count"] += 1
        elif isinstance(node, ast.If):
            features["branch_count"] += 1
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)):
            features["math_ops"] += 1
        elif isinstance(node, ast.Call):
            features["call_count"] += 1
        elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            features["comprehension_count"] += 1

    features["nesting_depth"] = count_nesting(tree)

    return features


def compute_diversity(features1: dict, features2: dict) -> float:
    """Compute diversity between two code samples based on their features."""
    keys = set(features1.keys()) | set(features2.keys())

    diff_sum = 0
    for k in keys:
        v1 = features1.get(k, 0)
        v2 = features2.get(k, 0)
        # Normalized difference
        max_val = max(abs(v1), abs(v2), 1)
        diff_sum += abs(v1 - v2) / max_val

    return diff_sum / len(keys) if keys else 0


# --- DSPy Signatures ---
class MutationSignature(dspy.Signature):
    """Generate an improved version of code using SEARCH/REPLACE diffs.

    You are given a parent solution and one inspiration solution. Mutate the parent
    code to improve its score, optionally borrowing ideas from the inspiration.
    Output SEARCH/REPLACE blocks to make surgical edits to the parent code.
    """

    problem_description: str = dspy.InputField(desc="The optimization problem description")
    function_signature: str = dspy.InputField(desc="The exact function signature to implement")
    parent_code: str = dspy.InputField(desc="The parent code to mutate (v1)")
    parent_score: float = dspy.InputField(desc="Score of the parent code")
    inspiration_code: str = dspy.InputField(desc="An inspiration solution to optionally borrow ideas from (v2)")
    inspiration_score: float = dspy.InputField(desc="Score of the inspiration code")

    diff_output: str = dspy.OutputField(desc="SEARCH/REPLACE blocks to modify the parent code")


class ParadigmShiftSignature(dspy.Signature):
    """Generate a fundamentally different algorithmic approach.

    You are given 3 representative solutions from different behavioral regions of
    the search space. Your task is to create a COMPLETELY DIFFERENT algorithmic
    approach that doesn't resemble any of them. The new approach should achieve
    a high score while being structurally and conceptually distinct.
    """

    problem_description: str = dspy.InputField(desc="The optimization problem description")
    function_signature: str = dspy.InputField(desc="The exact function signature to implement")
    n_evaluations: int = dspy.InputField(desc="Number of evaluations so far")
    representative_solutions: str = dspy.InputField(desc="3 best solutions from different behavioral regions")

    code: str = dspy.OutputField(desc="Complete Python code implementing a novel approach different from all 3 representatives")


# --- DSPy Modules ---
class MutationModule(dspy.Module):
    """Module for generating code mutations with parent + 1 inspiration."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(MutationSignature)

    def forward(self, problem_description: str, function_signature: str,
                parent_code: str, parent_score: float,
                inspiration_code: str, inspiration_score: float) -> dspy.Prediction:
        return self.generate(
            problem_description=problem_description,
            function_signature=function_signature,
            parent_code=parent_code,
            parent_score=parent_score,
            inspiration_code=inspiration_code,
            inspiration_score=inspiration_score,
        )


class ParadigmShiftModule(dspy.Module):
    """Module for generating paradigm shift solutions."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(ParadigmShiftSignature)

    def forward(self, problem_description: str, function_signature: str,
                n_evaluations: int, representative_solutions: str) -> dspy.Prediction:
        return self.generate(
            problem_description=problem_description,
            function_signature=function_signature,
            n_evaluations=n_evaluations,
            representative_solutions=representative_solutions,
        )


# --- Metrics ---
@dataclass
class MutationMetricResult:
    """Result from mutation metric evaluation."""
    diff_works: bool = False
    compiles: bool = False
    runs: bool = False
    score_improved: bool = False
    parent_score: float = 0.0
    child_score: float = 0.0
    improvement: float = 0.0


def mutation_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    inputs: list,
    trace: Optional[Any] = None,
    debug: bool = False,
) -> float:
    """
    Evaluate a mutation with partial credit at each stage:
    - Has diff structure (SEARCH/REPLACE blocks): 0.1
    - Diff applies correctly: +0.2
    - Code compiles: +0.2
    - Code runs without error: +0.2
    - Score maintained/improved: +0.3
    """
    parent_code = example.parent_code
    diff_output = prediction.diff_output
    score = 0.0

    # Check if output has diff structure
    has_diff_structure = "<<<<<<< SEARCH" in diff_output and "=======" in diff_output
    has_code_block = "```python" in diff_output or "```\n" in diff_output

    if has_diff_structure or has_code_block:
        score += 0.1  # At least tried to output code/diff
        if debug:
            print(f"  [METRIC] Has structure: diff={has_diff_structure}, code={has_code_block}")

    # Apply diff
    mutated_code = apply_diff(parent_code, diff_output)
    if mutated_code is None:
        if debug:
            print(f"  [METRIC] Diff failed to apply. Output length: {len(diff_output)}")
        return score  # Return partial credit

    score += 0.2  # Diff applied
    if debug:
        print(f"  [METRIC] Diff applied successfully")

    # Check compilation
    compiles, error = validate_code(mutated_code)
    if not compiles:
        if debug:
            print(f"  [METRIC] Code doesn't compile: {error}")
        return score  # Return partial credit

    score += 0.2  # Compiles
    if debug:
        print(f"  [METRIC] Code compiles")

    # Execute and check score
    result = execute_code(mutated_code, inputs)
    if "error" in result:
        if debug:
            print(f"  [METRIC] Execution error: {result['error']}")
        return score  # Return partial credit

    score += 0.2  # Runs without error
    child_score = result.get("score", 0.0)
    parent_score = example.parent_score

    if debug:
        print(f"  [METRIC] Runs OK. Parent: {parent_score:.2f}, Child: {child_score:.2f}")

    # Score improvement/maintenance component (up to 0.3)
    if child_score > parent_score:
        improvement = (child_score - parent_score) / max(parent_score, 1.0)
        improvement_score = min(improvement * 2, 1.0) * 0.3  # Cap at 0.3
        score += improvement_score
        if debug:
            print(f"  [METRIC] Score improved! +{improvement_score:.2f}")
    elif child_score >= parent_score * 0.95:
        score += 0.15  # Maintained score
        if debug:
            print(f"  [METRIC] Score maintained +0.15")
    elif child_score >= parent_score * 0.8:
        score += 0.05  # Slight regression but still decent
        if debug:
            print(f"  [METRIC] Slight regression +0.05")

    return score


@dataclass
class ParadigmMetricResult:
    """Result from paradigm shift metric evaluation."""
    compiles: bool = False
    runs: bool = False
    score: float = 0.0
    diversity: float = 0.0


def paradigm_shift_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    inputs: list,
    reference_solutions: list[str],
    trace: Optional[Any] = None,
) -> float:
    """
    Evaluate a paradigm shift:
    - Code compiles and runs (required, else 0)
    - High score (60%)
    - Different approach from references (40%)
    """
    code = extract_code(prediction.code) or prediction.code

    # Check compilation - MUST compile, else 0
    compiles, error = validate_code(code)
    if not compiles:
        return 0.0

    # Execute and get score - MUST run, else 0
    result = execute_code(code, inputs)
    if "error" in result:
        return 0.0

    exec_score = result.get("score", 0.0)

    # Score component (up to 60%)
    # Normalize score: assume range 0-90 for EPLB
    normalized_score = min(exec_score / 90.0, 1.0)
    score = normalized_score * 0.6

    # Diversity component (up to 40%)
    if reference_solutions:
        new_features = compute_code_features(code)
        diversities = []
        for ref_code in reference_solutions:
            ref_features = compute_code_features(ref_code)
            div = compute_diversity(new_features, ref_features)
            diversities.append(div)

        avg_diversity = sum(diversities) / len(diversities)
        # Normalize diversity (0-1 range)
        diversity_score = min(avg_diversity * 2, 1.0) * 0.4
        score += diversity_score

    return score


# --- Training Data Generation ---

# Alternative seed programs to use as inspirations
INSPIRATION_PROGRAM_1 = '''"""Simple greedy expert replication."""
import torch

def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy load-proportional replication."""
    num_layers, num_logical = weight.shape
    device = weight.device

    # Normalize weights to get replica counts
    weight_sum = weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
    normalized = weight / weight_sum

    # Allocate replicas proportional to load (minimum 1 each)
    expert_count = torch.ones(num_layers, num_logical, dtype=torch.int64, device=device)
    remaining = num_replicas - num_logical

    for _ in range(remaining):
        # Give extra replica to expert with highest load/count ratio
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

    # Pad log2phy to same shape
    max_replicas = expert_count.max().item()
    log2phy = torch.full((num_layers, num_logical, max_replicas), -1, dtype=torch.int64, device=device)
    for layer in range(num_layers):
        for expert in range(num_logical):
            slots = log2phy_list[layer][expert]
            log2phy[layer, expert, :len(slots)] = torch.tensor(slots, dtype=torch.int64)

    return phy2log, log2phy, expert_count
'''

INSPIRATION_PROGRAM_2 = '''"""Round-robin with load-aware adjustment."""
import torch

def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Round-robin assignment with load balancing."""
    num_layers, num_logical = weight.shape
    device = weight.device
    experts_per_gpu = num_replicas // num_gpus

    # Start with 1 replica per expert
    expert_count = torch.ones(num_layers, num_logical, dtype=torch.int64, device=device)
    extra = num_replicas - num_logical

    # Sort experts by weight and give extras to heaviest
    sorted_idx = weight.argsort(dim=1, descending=True)
    for layer in range(num_layers):
        for i in range(extra):
            expert_count[layer, sorted_idx[layer, i % num_logical]] += 1

    # Assign to physical slots round-robin across GPUs
    phy2log = torch.zeros(num_layers, num_replicas, dtype=torch.int64, device=device)

    for layer in range(num_layers):
        slot = 0
        for expert in range(num_logical):
            for _ in range(expert_count[layer, expert].item()):
                phy2log[layer, slot] = expert
                slot += 1

    # Build log2phy
    max_count = expert_count.max().item()
    log2phy = torch.full((num_layers, num_logical, max_count), -1, dtype=torch.int64, device=device)

    for layer in range(num_layers):
        counts = torch.zeros(num_logical, dtype=torch.int64)
        for slot in range(num_replicas):
            expert = phy2log[layer, slot].item()
            log2phy[layer, expert, counts[expert]] = slot
            counts[expert] += 1

    return phy2log, log2phy, expert_count
'''


def generate_mutation_examples(n_examples: int = 10) -> list[dspy.Example]:
    """Generate training examples for mutation optimization (parent + 1 inspiration)."""
    examples = []

    inputs = get_lazy_inputs()

    # Get scores for seed and inspirations
    parent_code = SEED_PROGRAM
    result = execute_code(parent_code, inputs)
    parent_score = result.get("score", 0.0) if "error" not in result else 50.0

    inspirations = [
        (INSPIRATION_PROGRAM_1, None),
        (INSPIRATION_PROGRAM_2, None),
    ]
    # Compute inspiration scores
    for i, (code, _) in enumerate(inspirations):
        result = execute_code(code, inputs)
        score = result.get("score", 0.0) if "error" not in result else 40.0
        inspirations[i] = (code, score)

    for i in range(n_examples):
        # Rotate through inspirations
        insp_code, insp_score = inspirations[i % len(inspirations)]

        example = dspy.Example(
            problem_description=PROBLEM_DESCRIPTION,
            function_signature=FUNCTION_SIGNATURE,
            parent_code=parent_code,
            parent_score=parent_score,
            inspiration_code=insp_code,
            inspiration_score=insp_score,
        ).with_inputs(
            "problem_description", "function_signature",
            "parent_code", "parent_score",
            "inspiration_code", "inspiration_score"
        )
        examples.append(example)

    return examples


def generate_paradigm_shift_examples(n_examples: int = 5) -> list[dspy.Example]:
    """Generate training examples for paradigm shift optimization (3 representative solutions)."""
    examples = []

    inputs = get_lazy_inputs()

    # Get scores for all representative solutions
    solutions = [
        ("Region 1 - Hierarchical Packing", SEED_PROGRAM),
        ("Region 2 - Greedy Load-Proportional", INSPIRATION_PROGRAM_1),
        ("Region 3 - Round-Robin Weighted", INSPIRATION_PROGRAM_2),
    ]

    scored_solutions = []
    for name, code in solutions:
        result = execute_code(code, inputs)
        score = result.get("score", 0.0) if "error" not in result else 40.0
        scored_solutions.append((name, code, score))

    # Format as representative solutions string
    representative_solutions = ""
    for i, (name, code, score) in enumerate(scored_solutions, 1):
        representative_solutions += f"""### {name} (Score: {score:.1f})
```python
{code}
```

"""

    for i in range(n_examples):
        example = dspy.Example(
            problem_description=PROBLEM_DESCRIPTION,
            function_signature=FUNCTION_SIGNATURE,
            n_evaluations=100 + i * 50,
            representative_solutions=representative_solutions,
        ).with_inputs("problem_description", "function_signature", "n_evaluations", "representative_solutions")
        examples.append(example)

    return examples


# --- Optimizer Setup ---
def setup_light_model_lm() -> dspy.LM:
    """Set up DSPy language model for light model (local vLLM)."""
    return dspy.LM(
        model=f"openai/{LIGHT_MODEL}",
        api_base=LIGHT_MODEL_ENDPOINT,
        api_key="dummy",  # vLLM doesn't require key
        temperature=0.8,
        max_tokens=8192,  # Increased from 4096 to avoid truncation
    )


def setup_heavy_model_lm() -> dspy.LM:
    """Set up DSPy language model for heavy model (OpenRouter)."""
    return dspy.LM(
        model=f"openrouter/{HEAVY_MODEL}",
        api_base=HEAVY_MODEL_ENDPOINT,
        api_key=OPENROUTER_API_KEY,
        temperature=0.7,
        max_tokens=8192,
    )


def optimize_mutation_prompt(
    n_trials: int = 20,
    n_examples: int = 10,
    output_file: Optional[str] = None,
    optimizer_type: str = "mipro",
    breadth: int = 5,
    depth: int = 2,
) -> dict:
    """
    Optimize the mutation generation prompt using COPRO or MIPROv2.

    Returns optimized instructions and metrics.
    """
    print("=" * 60)
    print("Optimizing MUTATION prompt")
    print("=" * 60)

    # Setup
    lm = setup_light_model_lm()
    dspy.configure(lm=lm)

    inputs = get_lazy_inputs()

    # Create metric that captures inputs
    def metric(example, prediction, trace=None, debug=False):
        return mutation_metric(example, prediction, inputs, trace, debug=debug)

    # Generate training examples
    trainset = generate_mutation_examples(n_examples)

    # Create module
    module = MutationModule()

    if optimizer_type == "copro":
        from dspy.teleprompt import COPRO

        optimizer = COPRO(
            metric=metric,
            breadth=breadth,
            depth=depth,
            init_temperature=0.8,
            verbose=True,
        )

        print(f"\nRunning COPRO optimization...")
        print(f"Training examples: {len(trainset)}, breadth={breadth}, depth={depth}")

        # COPRO compile
        optimized_module = optimizer.compile(
            module,
            trainset=trainset,
            eval_kwargs={"num_threads": 8},
        )
    else:
        # MIPROv2
        from dspy.teleprompt import MIPROv2

        optimizer = MIPROv2(
            metric=metric,
            auto=None,  # Disable auto mode
            num_candidates=5,  # Instruction candidates to try
            verbose=True,
            num_threads=8,  # Parallel evaluation threads (vLLM supports parallel)
        )

        print(f"\nRunning MIPROv2 optimization (zero-shot)...")
        print(f"Training examples: {len(trainset)}")

        # Optimize (zero-shot: instruction-only, no few-shot examples in proposal or output)
        optimized_module = optimizer.compile(
            module,
            trainset=trainset,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            num_trials=n_trials,
            minibatch=False,  # Don't use minibatching (small dataset)
            fewshot_aware_proposer=False,  # Don't include examples in instruction proposal (saves context)
        )

    # Extract optimized instructions
    optimized_instructions = {}
    for name, predictor in optimized_module.named_predictors():
        if hasattr(predictor, "signature"):
            optimized_instructions[name] = str(predictor.signature.instructions)

    # Evaluate on test examples
    testset = generate_mutation_examples(5)
    scores = []
    for example in testset:
        try:
            pred = optimized_module(**example.inputs())  # Use __call__ instead of forward
            score = metric(example, pred, debug=True)
            scores.append(score)
        except Exception as e:
            print(f"  Evaluation error: {e}")
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    result = {
        "target": "mutation",
        "optimized_instructions": optimized_instructions,
        "avg_score": avg_score,
        "n_trials": n_trials,
        "n_examples": n_examples,
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


def optimize_paradigm_shift_prompt(
    n_trials: int = 10,
    n_examples: int = 5,
    output_file: Optional[str] = None,
) -> dict:
    """
    Optimize the paradigm shift prompt using COPRO.

    Returns optimized instructions and metrics.
    """
    print("=" * 60)
    print("Optimizing PARADIGM SHIFT prompt")
    print("=" * 60)

    # Setup
    lm = setup_heavy_model_lm()
    dspy.configure(lm=lm)

    inputs = get_lazy_inputs()
    # Use all 3 programs as references for diversity measurement
    reference_solutions = [SEED_PROGRAM, INSPIRATION_PROGRAM_1, INSPIRATION_PROGRAM_2]

    # Create metric that captures inputs and references
    def metric(example, prediction, trace=None):
        return paradigm_shift_metric(example, prediction, inputs, reference_solutions, trace)

    # Generate training examples
    trainset = generate_paradigm_shift_examples(n_examples)

    # Create module
    module = ParadigmShiftModule()

    # Setup MIPROv2 optimizer
    from dspy.teleprompt import MIPROv2

    optimizer = MIPROv2(
        metric=metric,
        auto=None,  # Disable auto mode
        num_candidates=3,  # Instruction candidates
        verbose=True,
        num_threads=4,  # Parallel evaluation
    )

    print(f"\nRunning MIPROv2 optimization (zero-shot)...")
    print(f"Training examples: {len(trainset)}")

    # Optimize (zero-shot: instruction-only, no few-shot examples in proposal or output)
    optimized_module = optimizer.compile(
        module,
        trainset=trainset,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        num_trials=n_trials,
        minibatch=False,  # Don't use minibatching (small dataset)
        fewshot_aware_proposer=False,  # Don't include examples in instruction proposal (saves context)
    )

    # Extract optimized instructions
    optimized_instructions = {}
    for name, predictor in optimized_module.named_predictors():
        if hasattr(predictor, "signature"):
            optimized_instructions[name] = str(predictor.signature.instructions)

    # Evaluate on test examples
    testset = generate_paradigm_shift_examples(3)
    scores = []
    for example in testset:
        try:
            pred = optimized_module(**example.inputs())  # Use __call__ instead of forward
            score = metric(example, pred)
            scores.append(score)
        except Exception as e:
            print(f"  Evaluation error: {e}")
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    result = {
        "target": "paradigm_shift",
        "optimized_instructions": optimized_instructions,
        "avg_score": avg_score,
        "n_trials": n_trials,
        "n_examples": n_examples,
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


def update_prompt_files(mutation_result: Optional[dict], paradigm_result: Optional[dict], dry_run: bool = False):
    """Update the actual prompt files with optimized instructions."""
    import shutil

    prompts_file = Path(__file__).parent.parent.parent / "algoforge" / "equilibrium" / "prompts.py"
    builder_file = Path(__file__).parent.parent.parent / "algoforge" / "llm" / "prompts" / "builder.py"

    if paradigm_result and paradigm_result.get("optimized_instructions"):
        instructions = paradigm_result["optimized_instructions"]
        print("\n" + "=" * 60)
        print("PARADIGM SHIFT - Optimized Instructions")
        print("=" * 60)

        # Get the optimized instruction text
        instr_text = list(instructions.values())[0] if instructions else ""
        print(f"\n{instr_text}")

        if not dry_run and instr_text:
            # Backup original
            backup_path = prompts_file.with_suffix('.py.backup')
            shutil.copy(prompts_file, backup_path)
            print(f"\nBacked up to: {backup_path}")

            # Read and update the file
            content = prompts_file.read_text()

            # Find and replace the "Your Challenge" section with optimized instructions
            # We'll add the optimized instructions after the representative solutions section
            old_challenge = '''## Your Challenge: PARADIGM SHIFT

Generate a **fundamentally different algorithmic approach**.

### Instructions:
1. Study the function signature carefully - match it EXACTLY
2. Identify what approaches the existing solutions use
3. Design a solution using a COMPLETELY DIFFERENT strategy'''

            new_challenge = f'''## Your Challenge: PARADIGM SHIFT

{instr_text}

### Instructions:
1. Study the function signature carefully - match it EXACTLY
2. Identify what approaches the existing solutions use
3. Design a solution using a COMPLETELY DIFFERENT strategy'''

            if old_challenge in content:
                content = content.replace(old_challenge, new_challenge)
                prompts_file.write_text(content)
                print(f"Updated: {prompts_file}")
            else:
                print(f"WARNING: Could not find target section in {prompts_file}")

    if mutation_result and mutation_result.get("optimized_instructions"):
        instructions = mutation_result["optimized_instructions"]
        print("\n" + "=" * 60)
        print("MUTATION - Optimized Instructions")
        print("=" * 60)

        instr_text = list(instructions.values())[0] if instructions else ""
        print(f"\n{instr_text}")

        if not dry_run and instr_text:
            # Backup original
            backup_path = builder_file.with_suffix('.py.backup')
            shutil.copy(builder_file, backup_path)
            print(f"\nBacked up to: {backup_path}")

            # Read and update the file - update the DIFF mode instructions
            content = builder_file.read_text()

            # The key instruction to optimize is in _output_instructions for DIFF mode
            old_diff_intro = "Output your improved code using SEARCH/REPLACE blocks."
            new_diff_intro = instr_text.split('\n')[0] if instr_text else old_diff_intro

            if old_diff_intro in content:
                content = content.replace(old_diff_intro, new_diff_intro)
                builder_file.write_text(content)
                print(f"Updated: {builder_file}")
            else:
                print(f"WARNING: Could not find target section in {builder_file}")


# --- Main ---
def main():
    parser = argparse.ArgumentParser(
        description="Optimize AlgoForge prompts using DSPy COPRO"
    )
    parser.add_argument(
        "--target",
        choices=["mutation", "paradigm", "both"],
        default="both",
        help="Which prompt(s) to optimize",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=10,
        help="Number of training examples",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prompt_optimization_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't update prompt files, just show what would be changed",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update the prompt files (creates backups)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["mipro", "copro"],
        default="mipro",
        help="Which optimizer to use (mipro or copro)",
    )
    parser.add_argument(
        "--breadth",
        type=int,
        default=5,
        help="COPRO: number of candidate instructions per iteration",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="COPRO: number of refinement iterations",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mutation_result = None
    paradigm_result = None

    if args.target in ["mutation", "both"]:
        output_file = output_dir / f"mutation_optimized_{timestamp}.json"
        mutation_result = optimize_mutation_prompt(
            n_trials=args.n_trials,
            n_examples=args.n_examples,
            output_file=str(output_file),
            optimizer_type=args.optimizer,
            breadth=args.breadth,
            depth=args.depth,
        )

    if args.target in ["paradigm", "both"]:
        output_file = output_dir / f"paradigm_optimized_{timestamp}.json"
        paradigm_result = optimize_paradigm_shift_prompt(
            n_trials=args.n_trials // 2,  # Fewer trials for expensive model
            n_examples=args.n_examples // 2,
            output_file=str(output_file),
        )

    # Show summary and optionally update files
    dry_run = not args.apply
    if dry_run:
        print("\n[DRY RUN - use --apply to actually update prompt files]")
    update_prompt_files(mutation_result, paradigm_result, dry_run=dry_run)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    if dry_run:
        print("\nRun with --apply to update the prompt files")


if __name__ == "__main__":
    main()
