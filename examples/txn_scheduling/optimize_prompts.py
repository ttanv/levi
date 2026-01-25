#!/usr/bin/env python3
"""
DSPy prompt optimization for AlgoForge - Transaction Scheduling.

Optimizes prompts using MIPROv2:
1. Mutation prompts (one per model: MiMo, DeepSeek, Qwen)
2. Paradigm shift prompt (Gemini)

Budget: ~$0.60 for optimization, $4.40 for main run
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import ast

import dspy

# Add algoforge to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from problem import (
    PROBLEM_DESCRIPTION,
    FUNCTION_SIGNATURE,
    SEED_PROGRAM,
    INPUTS,
    score_fn,
)

# --- Model Configuration ---
LIGHT_MODELS = {
    "mimo": "openrouter/xiaomi/mimo-v2-flash",
    "deepseek": "openrouter/deepseek/deepseek-v3.2",
    "qwen": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}

LOCAL_ENDPOINTS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8001/v1",
}

PARADIGM_SHIFT_MODEL = "openrouter/google/gemini-3-flash-preview"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Behavioral features for TXN scheduling (matching TXN_AST_FEATURES)
TXN_AST_FEATURES = ['loop_nesting_max', 'comparison_count', 'call_count', 'branch_count']


# --- Code Utilities ---
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
    """Check if code compiles without errors."""
    try:
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, str(e)


def execute_code(code: str, inputs: list, timeout: int = 60) -> dict:
    """Execute code and return score_fn results with timeout."""
    import multiprocessing

    def _run(queue, code, inputs):
        namespace = {}
        try:
            exec(code, namespace)
            if "get_best_schedule" not in namespace:
                queue.put({"error": "Function 'get_best_schedule' not found"})
                return
            result = score_fn(namespace["get_best_schedule"], inputs)
            queue.put(result)
        except Exception as e:
            queue.put({"error": str(e)})

    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_run, args=(queue, code, inputs))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"error": "Execution timeout"}

    if queue.empty():
        return {"error": "No result returned"}

    return queue.get()


def compute_behavior_features(code: str) -> dict[str, float]:
    """Extract behavioral features matching TXN_AST_FEATURES."""
    features = {
        "loop_nesting_max": 0.0,
        "comparison_count": 0.0,
        "call_count": 0.0,
        "branch_count": 0.0,
    }

    try:
        tree = ast.parse(code)
    except:
        return features

    def count_nesting(node, depth=0):
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                child_depth = count_nesting(child, depth + 1)
            else:
                child_depth = count_nesting(child, depth)
            max_depth = max(max_depth, child_depth)
        return max_depth

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            depth = count_nesting(node)
            features["loop_nesting_max"] = max(features["loop_nesting_max"], depth)
        elif isinstance(node, ast.Compare):
            features["comparison_count"] += 1
        elif isinstance(node, ast.Call):
            features["call_count"] += 1
        elif isinstance(node, ast.If):
            features["branch_count"] += 1

    return features


def compute_diversity(features1: dict, features2: dict) -> float:
    """Compute normalized L1 distance between feature vectors."""
    keys = set(features1.keys()) | set(features2.keys())
    if not keys:
        return 0.0

    diff_sum = 0.0
    for k in keys:
        v1 = features1.get(k, 0)
        v2 = features2.get(k, 0)
        max_val = max(abs(v1), abs(v2), 1)
        diff_sum += abs(v1 - v2) / max_val

    return diff_sum / len(keys)


# --- DSPy Signatures ---
class MutationSignature(dspy.Signature):
    """Generate an improved version of code.

    You are given a parent solution and one inspiration solution. Mutate the parent
    code to improve its score, optionally borrowing ideas from the inspiration.
    Output the complete improved code.
    """

    problem_description: str = dspy.InputField(desc="The optimization problem description")
    function_signature: str = dspy.InputField(desc="The exact function signature to implement")
    parent_code: str = dspy.InputField(desc="The parent code to mutate (v1)")
    parent_score: float = dspy.InputField(desc="Score of the parent code")
    inspiration_code: str = dspy.InputField(desc="An inspiration solution to optionally borrow ideas from (v2)")
    inspiration_score: float = dspy.InputField(desc="Score of the inspiration code")

    code: str = dspy.OutputField(desc="Complete improved Python code")


class ParadigmShiftSignature(dspy.Signature):
    """Generate a high-scoring algorithmic solution using a fundamentally new approach.

    You are given representative solutions from different behavioral regions.
    First, carefully analyze each existing solution to understand:
    1. What algorithmic strategy it uses
    2. Why it achieves its current score
    3. Where it falls short or what limitations it has

    Then, synthesize these insights to propose a NEW solution that:
    - Learns from the strengths of existing approaches
    - Addresses the weaknesses and limitations you identified
    - Uses a fundamentally different algorithmic paradigm
    - Achieves a higher score by combining insights in a novel way

    Do not simply tweak or combine existing solutions. Propose a genuinely new
    approach that transcends the limitations of current solutions while
    incorporating the lessons learned from analyzing them.
    """

    problem_description: str = dspy.InputField(desc="The optimization problem description")
    function_signature: str = dspy.InputField(desc="The exact function signature to implement")
    best_score: float = dspy.InputField(desc="The best score achieved so far")
    representative_solutions: str = dspy.InputField(desc="Current solutions with their scores")

    code: str = dspy.OutputField(desc="Complete Python code that achieves a high score")


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
                best_score: float, representative_solutions: str) -> dspy.Prediction:
        return self.generate(
            problem_description=problem_description,
            function_signature=function_signature,
            best_score=best_score,
            representative_solutions=representative_solutions,
        )


# --- Metrics ---
def mutation_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    inputs: list,
    trace: Optional[Any] = None,
    debug: bool = False,
) -> float:
    """
    Evaluate a mutation with partial credit at each stage:
    - Has code block: 0.10
    - Code extracted: +0.15
    - Code compiles: +0.15
    - Code runs without error: +0.30 (HEAVY WEIGHT)
    - Score maintained/improved: +0.30
    """
    output = prediction.code
    score = 0.0

    # Check if output has code structure
    has_code_block = "```python" in output or "```\n" in output or "def " in output

    if has_code_block:
        score += 0.10
        if debug:
            print(f"  [METRIC] Has code structure")

    # Extract code
    mutated_code = extract_code(output)
    if mutated_code is None:
        # Try using raw output if it looks like code
        if "def " in output:
            mutated_code = output
        else:
            if debug:
                print(f"  [METRIC] Failed to extract code")
            return score

    score += 0.15
    if debug:
        print(f"  [METRIC] Code extracted")

    # Check compilation
    compiles, error = validate_code(mutated_code)
    if not compiles:
        if debug:
            print(f"  [METRIC] Code doesn't compile: {error}")
        return score

    score += 0.15
    if debug:
        print(f"  [METRIC] Code compiles")

    # Execute and check score
    result = execute_code(mutated_code, inputs)
    if "error" in result:
        if debug:
            print(f"  [METRIC] Execution error: {result['error'][:100]}")
        return score

    score += 0.30  # Runs without error (HEAVY WEIGHT)
    child_score = result.get("score", 0.0)
    parent_score = example.parent_score

    if debug:
        print(f"  [METRIC] Runs OK. Parent: {parent_score:.2f}, Child: {child_score:.2f}")

    # Score improvement/maintenance component (up to 0.30)
    if child_score > parent_score:
        improvement = (child_score - parent_score) / max(parent_score, 1.0)
        improvement_score = min(improvement * 2, 1.0) * 0.30
        score += improvement_score
        if debug:
            print(f"  [METRIC] Score improved! +{improvement_score:.2f}")
    elif child_score >= parent_score * 0.95:
        score += 0.15
        if debug:
            print(f"  [METRIC] Score maintained +0.15")
    elif child_score >= parent_score * 0.8:
        score += 0.05
        if debug:
            print(f"  [METRIC] Slight regression +0.05")

    return score


def paradigm_shift_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    inputs: list,
    reference_solutions: list[str],
    trace: Optional[Any] = None,
) -> float:
    """
    Evaluate a paradigm shift:
    - Hard gate: Must compile AND run (else 0)
    - Score: 80% (PRIMARY - maximize execution score)
    - Diversity: 20% (secondary - some novelty bonus)
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

    # Score component (80%) - normalize to 0-100 range
    normalized_score = min(exec_score / 100.0, 1.0)
    score = normalized_score * 0.80

    # Diversity component (20%)
    if reference_solutions:
        new_features = compute_behavior_features(code)
        diversities = []
        for ref_code in reference_solutions:
            ref_features = compute_behavior_features(ref_code)
            div = compute_diversity(new_features, ref_features)
            diversities.append(div)

        # Min distance to existing solutions
        min_diversity = min(diversities) if diversities else 0.0
        diversity_score = min(min_diversity * 2, 1.0) * 0.20
        score += diversity_score

    return score


# --- Training Data Generation ---
INSPIRATION_PROGRAM_1 = '''import random

def get_best_schedule(workload):
    """Conflict-aware greedy scheduling."""
    random.seed(42)
    n = workload.num_txns

    # Build conflict graph
    conflicts = [[0] * n for _ in range(n)]
    for i in range(n):
        keys_i = set(op[1] for op in workload.txns[i])
        writes_i = set(op[1] for op in workload.txns[i] if op[0] == 'w')
        for j in range(i + 1, n):
            keys_j = set(op[1] for op in workload.txns[j])
            writes_j = set(op[1] for op in workload.txns[j] if op[0] == 'w')
            # Conflict if both write same key or one writes what other reads
            if writes_i & keys_j or writes_j & keys_i:
                conflicts[i][j] = conflicts[j][i] = 1

    # Sort by total conflicts (ascending)
    conflict_counts = [sum(row) for row in conflicts]
    schedule = sorted(range(n), key=lambda x: conflict_counts[x])

    return workload.get_opt_seq_cost(schedule), schedule
'''

INSPIRATION_PROGRAM_2 = '''import random
import heapq

def get_best_schedule(workload):
    """Priority queue scheduling by transaction length."""
    random.seed(42)
    n = workload.num_txns

    # Priority by transaction length (shorter first)
    txn_lens = [(len(workload.txns[i]), i) for i in range(n)]
    heapq.heapify(txn_lens)

    schedule = []
    while txn_lens:
        _, txn = heapq.heappop(txn_lens)
        schedule.append(txn)

    return workload.get_opt_seq_cost(schedule), schedule
'''


def generate_mutation_examples(n_examples: int = 5) -> list[dspy.Example]:
    """Generate training examples for mutation optimization."""
    examples = []

    # Get scores for seed and inspirations
    parent_code = SEED_PROGRAM
    result = execute_code(parent_code, INPUTS)
    parent_score = result.get("score", 0.0) if "error" not in result else 20.0

    inspirations = [
        (INSPIRATION_PROGRAM_1, None),
        (INSPIRATION_PROGRAM_2, None),
    ]
    for i, (code, _) in enumerate(inspirations):
        result = execute_code(code, INPUTS)
        score = result.get("score", 0.0) if "error" not in result else 15.0
        inspirations[i] = (code, score)

    for i in range(n_examples):
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


def generate_paradigm_shift_examples(n_examples: int = 4) -> list[dspy.Example]:
    """Generate training examples for paradigm shift optimization."""
    examples = []

    solutions = [
        ("Region 1 - Greedy Cost Sampling", SEED_PROGRAM),
        ("Region 2 - Conflict-Aware Greedy", INSPIRATION_PROGRAM_1),
        ("Region 3 - Length-Priority Queue", INSPIRATION_PROGRAM_2),
    ]

    scored_solutions = []
    for name, code in solutions:
        result = execute_code(code, INPUTS)
        score = result.get("score", 0.0) if "error" not in result else 15.0
        scored_solutions.append((name, code, score))

    representative_solutions = ""
    for i, (name, code, score) in enumerate(scored_solutions, 1):
        representative_solutions += f"""### {name} (Score: {score:.1f})
```python
{code}
```

"""

    best_score = max(s[2] for s in scored_solutions)

    for i in range(n_examples):
        example = dspy.Example(
            problem_description=PROBLEM_DESCRIPTION,
            function_signature=FUNCTION_SIGNATURE,
            best_score=best_score,
            representative_solutions=representative_solutions,
        ).with_inputs("problem_description", "function_signature", "best_score", "representative_solutions")
        examples.append(example)

    return examples


# --- Optimizer Setup ---
def setup_model_lm(model: str) -> dspy.LM:
    """Set up DSPy language model for a given model."""
    if model in LOCAL_ENDPOINTS:
        return dspy.LM(
            model=f"openai/{model}",
            api_base=LOCAL_ENDPOINTS[model],
            api_key="dummy",
            temperature=0.8,
            max_tokens=8192,
        )
    else:
        # Strip openrouter/ prefix when using api_base directly
        model_id = model.replace("openrouter/", "")
        return dspy.LM(
            model=f"openai/{model_id}",
            api_base="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            temperature=0.8,
            max_tokens=8192,
        )


def optimize_mutation_for_model(
    model_key: str,
    model: str,
    n_trials: int = 10,
    n_examples: int = 5,
) -> dict:
    """Optimize mutation prompt for a specific model."""
    print(f"\n{'='*60}")
    print(f"Optimizing MUTATION for {model_key} ({model})")
    print(f"{'='*60}")

    lm = setup_model_lm(model)
    dspy.configure(lm=lm)

    def metric(example, prediction, trace=None):
        return mutation_metric(example, prediction, INPUTS, trace, debug=False)

    trainset = generate_mutation_examples(n_examples)
    module = MutationModule()

    from dspy.teleprompt import MIPROv2

    optimizer = MIPROv2(
        metric=metric,
        auto=None,
        num_candidates=3,
        verbose=True,
        num_threads=4,
    )

    print(f"Running MIPROv2: {len(trainset)} examples, {n_trials} trials")

    optimized_module = optimizer.compile(
        module,
        trainset=trainset,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        num_trials=n_trials,
        minibatch=False,
        fewshot_aware_proposer=False,
    )

    optimized_instructions = {}
    for name, predictor in optimized_module.named_predictors():
        if hasattr(predictor, "signature"):
            optimized_instructions[name] = str(predictor.signature.instructions)

    # Test evaluation
    testset = generate_mutation_examples(3)
    scores = []
    for example in testset:
        try:
            pred = optimized_module(**example.inputs())
            score = metric(example, pred)
            scores.append(score)
        except Exception as e:
            print(f"  Eval error: {e}")
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{model_key} optimization complete! Avg score: {avg_score:.3f}")

    return {
        "model_key": model_key,
        "model": model,
        "optimized_instructions": optimized_instructions,
        "avg_score": avg_score,
        "n_trials": n_trials,
    }


def optimize_paradigm_shift(
    n_trials: int = 8,
    n_examples: int = 4,
) -> dict:
    """Optimize paradigm shift prompt for Gemini."""
    print(f"\n{'='*60}")
    print(f"Optimizing PARADIGM SHIFT ({PARADIGM_SHIFT_MODEL})")
    print(f"{'='*60}")

    lm = setup_model_lm(PARADIGM_SHIFT_MODEL)
    dspy.configure(lm=lm)

    reference_solutions = [SEED_PROGRAM, INSPIRATION_PROGRAM_1, INSPIRATION_PROGRAM_2]

    def metric(example, prediction, trace=None):
        return paradigm_shift_metric(example, prediction, INPUTS, reference_solutions, trace)

    trainset = generate_paradigm_shift_examples(n_examples)
    module = ParadigmShiftModule()

    from dspy.teleprompt import MIPROv2

    optimizer = MIPROv2(
        metric=metric,
        auto=None,
        num_candidates=3,
        verbose=True,
        num_threads=2,
    )

    print(f"Running MIPROv2: {len(trainset)} examples, {n_trials} trials")

    optimized_module = optimizer.compile(
        module,
        trainset=trainset,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        num_trials=n_trials,
        minibatch=False,
        fewshot_aware_proposer=False,
    )

    optimized_instructions = {}
    for name, predictor in optimized_module.named_predictors():
        if hasattr(predictor, "signature"):
            optimized_instructions[name] = str(predictor.signature.instructions)

    # Test evaluation
    testset = generate_paradigm_shift_examples(2)
    scores = []
    for example in testset:
        try:
            pred = optimized_module(**example.inputs())
            score = metric(example, pred)
            scores.append(score)
        except Exception as e:
            print(f"  Eval error: {e}")
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\nParadigm shift optimization complete! Avg score: {avg_score:.3f}")

    return {
        "optimized_instructions": optimized_instructions,
        "avg_score": avg_score,
        "n_trials": n_trials,
    }


def run_optimization(budget: float = 0.60) -> dict:
    """
    Run full optimization within budget.

    Returns dict with optimized prompts per model.
    """
    print(f"\n{'#'*60}")
    print(f"# DSPy Prompt Optimization - TXN Scheduling")
    print(f"# Budget: ${budget:.2f}")
    print(f"{'#'*60}")

    results = {
        "mutation": {},
        "paradigm_shift": None,
        "timestamp": datetime.now().isoformat(),
        "budget": budget,
    }

    # Mutation: optimize for each model
    for model_key, model in LIGHT_MODELS.items():
        try:
            result = optimize_mutation_for_model(
                model_key=model_key,
                model=model,
                n_trials=10,
                n_examples=5,
            )
            results["mutation"][model_key] = result
        except Exception as e:
            print(f"Error optimizing mutation for {model_key}: {e}")
            results["mutation"][model_key] = {"error": str(e)}

    # Paradigm shift
    try:
        results["paradigm_shift"] = optimize_paradigm_shift(
            n_trials=8,
            n_examples=4,
        )
    except Exception as e:
        print(f"Error optimizing paradigm shift: {e}")
        results["paradigm_shift"] = {"error": str(e)}

    return results


def save_optimized_prompts(results: dict, output_file: str = "optimized_prompts.json"):
    """Save optimized prompts to JSON file."""
    output_path = Path(__file__).parent / output_file

    # Extract just the instructions for use in AlgoForge
    prompts = {
        "mutation": {},
        "paradigm_shift": None,
        "metadata": {
            "timestamp": results.get("timestamp"),
            "budget": results.get("budget"),
        }
    }

    for model_key, result in results.get("mutation", {}).items():
        if "optimized_instructions" in result:
            instructions = result["optimized_instructions"]
            # Get the first (and usually only) instruction
            if instructions:
                prompts["mutation"][model_key] = list(instructions.values())[0]

    if results.get("paradigm_shift") and "optimized_instructions" in results["paradigm_shift"]:
        instructions = results["paradigm_shift"]["optimized_instructions"]
        if instructions:
            prompts["paradigm_shift"] = list(instructions.values())[0]

    with open(output_path, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"\nOptimized prompts saved to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize prompts for TXN scheduling")
    parser.add_argument("--budget", type=float, default=0.60, help="Budget for optimization")
    parser.add_argument("--output", type=str, default="optimized_prompts.json", help="Output file")
    parser.add_argument("--paradigm-only", action="store_true", help="Only optimize paradigm shift prompt")
    parser.add_argument("--mutation-only", action="store_true", help="Only optimize mutation prompts")
    parser.add_argument("--trials", type=int, default=None, help="Override number of trials")
    parser.add_argument("--examples", type=int, default=None, help="Override number of examples")
    args = parser.parse_args()

    if args.paradigm_only:
        print("\n" + "#"*60)
        print("# DSPy Prompt Optimization - PARADIGM SHIFT ONLY")
        print("#"*60)

        n_trials = args.trials or 20
        n_examples = args.examples or 8

        result = optimize_paradigm_shift(n_trials=n_trials, n_examples=n_examples)

        # Save just paradigm shift
        prompts = {
            "mutation": {},
            "paradigm_shift": None,
            "metadata": {"timestamp": datetime.now().isoformat()}
        }
        if result and "optimized_instructions" in result:
            instructions = result["optimized_instructions"]
            if instructions:
                prompts["paradigm_shift"] = list(instructions.values())[0]

        output_path = Path(__file__).parent / args.output
        with open(output_path, "w") as f:
            json.dump(prompts, f, indent=2)
        print(f"\nSaved to: {output_path}")

    elif args.mutation_only:
        print("\n" + "#"*60)
        print("# DSPy Prompt Optimization - MUTATION ONLY")
        print("#"*60)

        n_trials = args.trials or 10
        n_examples = args.examples or 5

        results = {"mutation": {}, "paradigm_shift": None}
        for model_key, model in LIGHT_MODELS.items():
            try:
                result = optimize_mutation_for_model(
                    model_key=model_key,
                    model=model,
                    n_trials=n_trials,
                    n_examples=n_examples,
                )
                results["mutation"][model_key] = result
            except Exception as e:
                print(f"Error optimizing {model_key}: {e}")

        save_optimized_prompts(results, args.output)
    else:
        results = run_optimization(budget=args.budget)
        save_optimized_prompts(results, args.output)

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
