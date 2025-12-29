#!/usr/bin/env python3
"""
AlphaEvolve on Transaction Scheduling Problem.
"""

import asyncio
import sys
import time
from pathlib import Path
from concurrent.futures import BrokenExecutor

# Add algoforge to path (must be before algoforge imports)
ALGOFORGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ALGOFORGE_ROOT))

from algoforge.utils import ResilientProcessPool

# Add txn_scheduling resources to path
TXN_RESOURCES = ALGOFORGE_ROOT.parent / "ADRS-Leaderboard" / "problems" / "txn_scheduling" / "resources"
sys.path.insert(0, str(TXN_RESOURCES))

from algoforge.core import Program, EvaluationResult
from algoforge.pool import CVTMAPElitesPool
from algoforge.llm import PromptBuilder, ProgramWithScore, OutputMode
from algoforge.behavior.extractor import FeatureVector
import ast
import re
import json
import numpy as np


class TxnSchedulingBehaviorExtractor:
    """
    Custom behavior extractor for transaction scheduling with z-score normalization.

    Features (all normalized to ~[0, 1] via sigmoid of z-score):
    - execution_time: Wall clock time
    - loop_count: Number of for/while loops
    - branch_count: Number of if/elif/else branches
    - math_operators: Count of math ops (+, -, *, /, etc.)
    - workload_1_score: Performance on WORKLOAD_1 (already 0-1)
    - workload_2_score: Performance on WORKLOAD_2 (already 0-1)
    - workload_3_score: Performance on WORKLOAD_3 (already 0-1)
    """

    def __init__(self, max_time: float = 200.0):
        self.features = [
            'execution_time',
            'loop_count',
            'branch_count',
            'math_operators',
            'workload_1_score',
            'workload_2_score',
            'workload_3_score',
        ]
        self.max_time = max_time

        # Running statistics for z-score normalization (Welford's online algorithm)
        # Keys: feature names that need z-score normalization
        self._zscore_features = ['loop_count', 'branch_count', 'math_operators']
        self._count = 0
        self._mean = {f: 0.0 for f in self._zscore_features}
        self._M2 = {f: 0.0 for f in self._zscore_features}  # Sum of squared differences

    def _update_stats(self, feature: str, value: float):
        """Update running mean and variance using Welford's algorithm."""
        self._count += 1
        delta = value - self._mean[feature]
        self._mean[feature] += delta / self._count
        delta2 = value - self._mean[feature]
        self._M2[feature] += delta * delta2

    def _get_std(self, feature: str) -> float:
        """Get current standard deviation for a feature."""
        if self._count < 2:
            return 1.0  # Avoid division by zero
        variance = self._M2[feature] / (self._count - 1)
        return max(np.sqrt(variance), 0.1)  # Min std of 0.1 to avoid extreme z-scores

    def _zscore_to_01(self, z: float) -> float:
        """Convert z-score to [0, 1] using sigmoid."""
        # Sigmoid: 1 / (1 + exp(-z))
        # Clamp z to avoid overflow
        z = max(-10, min(10, z))
        return 1.0 / (1.0 + np.exp(-z))

    def extract(self, program: Program) -> FeatureVector:
        """Extract behavioral features from a program."""
        code = program.code
        metadata = program.metadata or {}
        values = {}

        # Parse AST once for static features
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return FeatureVector({f: 0.5 for f in self.features})

        # === Dynamic features (from evaluation metadata) ===

        # execution_time: normalize using max_time (0 = max_time, 1 = instant)
        raw_time = metadata.get('execution_time', self.max_time)
        values['execution_time'] = max(0.0, 1.0 - raw_time / self.max_time)

        # Per-workload scores: already 0-1 normalized
        values['workload_1_score'] = metadata.get('workload_1_score', 0.0)
        values['workload_2_score'] = metadata.get('workload_2_score', 0.0)
        values['workload_3_score'] = metadata.get('workload_3_score', 0.0)

        # === Static features (from code analysis, z-score normalized) ===

        # Count loops
        loop_count = sum(1 for node in ast.walk(tree)
                        if isinstance(node, (ast.For, ast.While)))

        # Count branches
        branch_count = sum(1 for node in ast.walk(tree)
                          if isinstance(node, ast.If))

        # Count math operators
        math_ops = sum(1 for node in ast.walk(tree)
                      if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.AugAssign)))

        # Update running statistics
        raw_values = {
            'loop_count': float(loop_count),
            'branch_count': float(branch_count),
            'math_operators': float(math_ops),
        }

        for feature in self._zscore_features:
            self._update_stats(feature, raw_values[feature])

        # Apply z-score normalization with sigmoid
        for feature in self._zscore_features:
            raw = raw_values[feature]
            z = (raw - self._mean[feature]) / self._get_std(feature)
            values[feature] = self._zscore_to_01(z)

        return FeatureVector(values)

def validate_python_syntax(code: str) -> tuple[bool, str]:
    """Check if code is valid Python syntax. Returns (is_valid, error_message)."""
    try:
        compile(code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def quick_validate_code(code: str) -> tuple[bool, str]:
    """Quick validation to catch obvious errors before full evaluation."""
    # Check for missing function definitions
    if 'def get_random_costs' not in code:
        return False, "Missing get_random_costs function"

    if 'def get_best_schedule' not in code:
        return False, "Missing get_best_schedule function"

    # Check for required imports
    if 'from txn_simulator import Workload' not in code and 'import Workload' not in code:
        return False, "Missing Workload import"

    # Quick execution test to check for basic errors
    try:
        import random
        import time as time_module
        import math
        import collections
        import heapq

        # We can't actually run the code without the workload data,
        # but we can at least compile and exec it to catch import errors
        namespace = {
            'time': time_module, 'random': random, 'np': np, 'numpy': np,
            'collections': collections, 'heapq': heapq, 'math': math,
            '__builtins__': __builtins__,
        }

        # Mock the workload imports for validation
        class MockWorkload:
            def __init__(self, data):
                self.num_txns = 100
                self.txns = [[(0, 0, 0, 10)] for _ in range(100)]
                self.debug = False
            def get_opt_seq_cost(self, seq):
                return len(seq) * 10

        namespace['Workload'] = MockWorkload
        namespace['WORKLOAD_1'] = {}
        namespace['WORKLOAD_2'] = {}
        namespace['WORKLOAD_3'] = {}

        exec(code, namespace)

        if 'get_random_costs' not in namespace:
            return False, "get_random_costs not defined after exec"
        if 'get_best_schedule' not in namespace:
            return False, "get_best_schedule not defined after exec"

        return True, ""
    except Exception as e:
        return False, f"Quick validation failed: {str(e)[:100]}"


def extract_and_validate_code(response: str) -> tuple[str | None, str]:
    """Extract and validate Python code from LLM response. Returns (code, error_msg)."""
    # Strip Qwen3 thinking tags (and similar patterns)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
    response = response.strip()

    candidates = []

    # Try to extract from ```python blocks
    code_pattern = r'```python\s*(.*?)\s*```'
    code_matches = re.findall(code_pattern, response, re.DOTALL)
    for match in code_matches:
        candidates.append(match.strip())

    # Try generic ``` blocks
    generic_pattern = r'```\s*(.*?)\s*```'
    generic_matches = re.findall(generic_pattern, response, re.DOTALL)
    for match in generic_matches:
        if 'def get_random_costs' in match or 'def get_best_schedule' in match:
            candidates.append(match.strip())

    # Try raw extraction if no code blocks
    if not candidates and ('def get_random_costs' in response or 'def get_best_schedule' in response):
        lines = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if 'import ' in line or 'from ' in line or 'def get_' in line:
                in_code = True
            if in_code:
                if line.strip().startswith('```'):
                    continue
                code_lines.append(line)
        if code_lines:
            candidates.append('\n'.join(code_lines).strip())

    # Validate candidates in order of length (prefer longer/more complete code)
    last_error = "No code found"
    for candidate in sorted(candidates, key=len, reverse=True):
        # Check syntax first
        is_valid, syntax_err = validate_python_syntax(candidate)
        if not is_valid:
            last_error = f"Syntax: {syntax_err}"
            continue

        # Quick validation
        is_valid, validation_err = quick_validate_code(candidate)
        if is_valid:
            return candidate, ""
        else:
            last_error = validation_err

    return None, last_error


from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3
from prompts import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, SEED_INSPIRATIONS, DIVERSITY_SEED_PROMPT, META_ADVISOR_PROMPT
import litellm
import logging

# Suppress LiteLLM debug spam
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
litellm.suppress_debug_info = True
litellm.set_verbose = False

# Register custom models not yet in litellm's mapping
litellm.register_model({
    "openrouter/google/gemini-2.5-flash-lite": {
        "max_tokens": 32768,
        "max_input_tokens": 1048576,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000004,
        "litellm_provider": "openrouter",
    },
    "openrouter/deepseek/deepseek-v3.2": {
        "max_tokens": 163840,
        "max_input_tokens": 163840,
        "max_output_tokens": 163840,
        "input_cost_per_token": 0.00000026,
        "output_cost_per_token": 0.00000038,
        "litellm_provider": "openrouter",
    },
    "openrouter/qwen/qwen3-coder-30b-a3b-instruct": {
        "max_tokens": 32768,
        "max_input_tokens": 160000,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.00000007,
        "output_cost_per_token": 0.00000027,
        "litellm_provider": "openrouter",
    },
    "openrouter/z-ai/glm-4.7": {
        "max_tokens": 32768,
        "max_input_tokens": 202752,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0000004,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "openrouter",
    },
})

# Pre-load workloads (lazy init for multiprocessing)
_WORKLOADS = None
_BASELINE = None
_EFFECTIVE_OPTIMAL = None
_PER_WORKLOAD_BASELINES = None
_PER_WORKLOAD_OPTIMALS = None


def _init_workloads():
    """Initialize workloads (called once per process)."""
    global _WORKLOADS, _BASELINE, _EFFECTIVE_OPTIMAL, _PER_WORKLOAD_BASELINES, _PER_WORKLOAD_OPTIMALS
    if _WORKLOADS is None:
        _WORKLOADS = [Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)]

        # Per-workload baselines and optimals
        _PER_WORKLOAD_BASELINES = []
        _PER_WORKLOAD_OPTIMALS = []
        for w in _WORKLOADS:
            baseline = w.get_opt_seq_cost(list(range(w.num_txns)))
            theoretical_optimal = max(txn[0][3] for txn in w.txns)
            effective_optimal = theoretical_optimal + 0.10 * (baseline - theoretical_optimal)
            _PER_WORKLOAD_BASELINES.append(baseline)
            _PER_WORKLOAD_OPTIMALS.append(effective_optimal)

        # Aggregate baselines
        _BASELINE = sum(_PER_WORKLOAD_BASELINES)
        _EFFECTIVE_OPTIMAL = sum(_PER_WORKLOAD_OPTIMALS)
    return _WORKLOADS, _BASELINE, _EFFECTIVE_OPTIMAL, _PER_WORKLOAD_BASELINES, _PER_WORKLOAD_OPTIMALS


class ConstraintEnforcingWorkload:
    """Wrapper around Workload that enforces O(n) constraint on get_opt_seq_cost calls."""

    # Class-level call counter shared across all instances
    _total_calls = 0
    _max_allowed_calls = 0
    _constraint_violated = False

    @classmethod
    def reset_counters(cls, max_calls: int):
        """Reset counters before evaluation."""
        cls._total_calls = 0
        cls._max_allowed_calls = max_calls
        cls._constraint_violated = False

    @classmethod
    def get_stats(cls):
        """Get call statistics."""
        return {
            "total_calls": cls._total_calls,
            "max_allowed": cls._max_allowed_calls,
            "violated": cls._constraint_violated,
        }

    def __init__(self, workload_data, original_workload_class):
        """Wrap the original workload."""
        self._inner = original_workload_class(workload_data)
        self._local_calls = 0

    @property
    def num_txns(self):
        return self._inner.num_txns

    @property
    def txns(self):
        return self._inner.txns

    @property
    def debug(self):
        return self._inner.debug

    @debug.setter
    def debug(self, value):
        self._inner.debug = value

    def get_opt_seq_cost(self, seq):
        """Track calls and enforce O(n) constraint."""
        ConstraintEnforcingWorkload._total_calls += 1
        self._local_calls += 1

        # Check if we've exceeded the O(n) budget
        if ConstraintEnforcingWorkload._total_calls > ConstraintEnforcingWorkload._max_allowed_calls:
            ConstraintEnforcingWorkload._constraint_violated = True
            raise RuntimeError(
                f"O(n) CONSTRAINT VIOLATED: get_opt_seq_cost called {ConstraintEnforcingWorkload._total_calls} times, "
                f"max allowed is {ConstraintEnforcingWorkload._max_allowed_calls} (2*total_txns). "
                f"Your algorithm must be O(n), not O(n²)."
            )

        return self._inner.get_opt_seq_cost(seq)


def compute_workload_score(makespan: float, baseline: float, effective_optimal: float) -> float:
    """Compute 0-1 normalized score for a single workload."""
    if makespan >= baseline:
        return 0.0
    if makespan <= effective_optimal:
        return 1.0
    return (baseline - makespan) / (baseline - effective_optimal)


def evaluate_code_in_process(code: str) -> dict:
    """Execute scheduling code in a subprocess. Must be picklable."""
    import random
    import numpy as np
    import collections
    import heapq
    import math
    import time as time_module

    # Ensure paths are set up in worker process
    import sys
    from pathlib import Path
    algoforge_root = Path(__file__).resolve().parents[2]
    txn_resources = algoforge_root.parent / "ADRS-Leaderboard" / "problems" / "txn_scheduling" / "resources"
    if str(algoforge_root) not in sys.path:
        sys.path.insert(0, str(algoforge_root))
    if str(txn_resources) not in sys.path:
        sys.path.insert(0, str(txn_resources))

    from txn_simulator import Workload as OriginalWorkload
    from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

    workloads, baseline, _, per_workload_baselines, per_workload_optimals = _init_workloads()

    # Calculate O(n) budget: allow 20*n calls (seed program uses ~10*n for sampling greedy)
    # Sum actual num_txns from each workload
    total_txns = sum(w.num_txns for w in workloads)
    max_calls_budget = 20 * total_txns  # 20*n total across all workloads (conservative for O(n) algo)

    # Reset constraint tracking
    ConstraintEnforcingWorkload.reset_counters(max_calls_budget)

    # Create a factory that wraps workloads with constraint enforcement
    def ConstrainedWorkloadFactory(workload_data):
        return ConstraintEnforcingWorkload(workload_data, OriginalWorkload)

    namespace = {
        'Workload': ConstrainedWorkloadFactory, 'WORKLOAD_1': WORKLOAD_1,
        'WORKLOAD_2': WORKLOAD_2, 'WORKLOAD_3': WORKLOAD_3,
        'time': time_module, 'random': random, 'np': np, 'numpy': np,
        'collections': collections, 'heapq': heapq, 'math': math,
        '__builtins__': __builtins__,
    }

    try:
        exec(code, namespace)
        if 'get_random_costs' not in namespace:
            return {"error": "get_random_costs not defined"}

        eval_start = time_module.time()
        result = namespace['get_random_costs']()
        wall_time = time_module.time() - eval_start

        # Handle both old (3-tuple) and new (4-tuple) return formats
        if len(result) == 4:
            makespan, schedules, per_workload_makespans, algo_time = result
        else:
            # Legacy format without per-workload costs
            makespan, schedules, algo_time = result
            per_workload_makespans = None

        # Get constraint stats
        constraint_stats = ConstraintEnforcingWorkload.get_stats()

        # Validate schedules
        for i, sched in enumerate(schedules):
            if set(sched) != set(range(workloads[i].num_txns)):
                return {"error": f"Invalid schedule for workload {i}"}

        # Compute per-workload scores from returned makespans (no recomputation!)
        per_workload_scores = []
        if per_workload_makespans is not None:
            for i, wl_makespan in enumerate(per_workload_makespans):
                wl_score = compute_workload_score(
                    wl_makespan,
                    per_workload_baselines[i],
                    per_workload_optimals[i]
                )
                per_workload_scores.append(wl_score)
        else:
            # Fallback: set scores to 0 if not provided (legacy code)
            per_workload_makespans = [0, 0, 0]
            per_workload_scores = [0.0, 0.0, 0.0]

        return {
            "makespan": makespan,
            "algo_time": algo_time,
            "wall_time": wall_time,
            "get_opt_seq_cost_calls": constraint_stats["total_calls"],
            "max_allowed_calls": constraint_stats["max_allowed"],
            # Per-workload metrics
            "workload_1_makespan": per_workload_makespans[0],
            "workload_2_makespan": per_workload_makespans[1],
            "workload_3_makespan": per_workload_makespans[2],
            "workload_1_score": per_workload_scores[0],  # 0-1 normalized
            "workload_2_score": per_workload_scores[1],  # 0-1 normalized
            "workload_3_score": per_workload_scores[2],  # 0-1 normalized
        }
    except RuntimeError as e:
        # Catch O(n) constraint violations
        if "O(n) CONSTRAINT VIOLATED" in str(e):
            return {"error": str(e)}
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


# Initialize in main process
WORKLOADS, BASELINE, EFFECTIVE_OPTIMAL = None, None, None


def compute_score(makespan: float) -> float:
    """Compute 0-100 raw score using leaderboard formula."""
    if makespan >= BASELINE:
        return 0.0
    if makespan <= EFFECTIVE_OPTIMAL:
        return 100.0
    return ((BASELINE - makespan) / (BASELINE - EFFECTIVE_OPTIMAL)) * 100


def score_fn(output: dict) -> float:
    """Score using leaderboard formula."""
    if output is None or "error" in output:
        return 0.0
    return compute_score(output.get("makespan", BASELINE))




# Budget
BUDGET_USD = 3.0

# State tracking
STATE_FILE = Path(__file__).parent / "evolution_state.json"


def save_state(
    generation: int,
    pool: CVTMAPElitesPool,
    best_score: float,
    best_program: Program,
    total_cost: float,
    extra_info: dict = None,
):
    """Save current evolution state to JSON file."""
    # Get archive elites
    elites = []
    try:
        # Access internal elites from the pool
        for cell_idx, elite in pool._elites.items():
            elite_info = {
                "cell_index": int(cell_idx),
                "score": float(elite.result.primary_score),
                "code": elite.program.code,
                "metadata": {k: float(v) if isinstance(v, (int, float)) else str(v)
                            for k, v in (elite.program.metadata or {}).items()},
            }
            # Add behavior vector if available
            if elite.behavior:
                elite_info["behavior"] = {k: float(v) for k, v in elite.behavior.values.items()}
            elites.append(elite_info)
    except AttributeError:
        # Pool doesn't expose _elites directly
        pass

    # Sort elites by score descending
    elites.sort(key=lambda x: x["score"], reverse=True)

    state = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generation": generation,
        "archive_size": pool.size(),
        "best_score": float(best_score),
        "total_cost_usd": float(total_cost),
        "best_program": {
            "code": best_program.code,
            "metadata": {k: float(v) if isinstance(v, (int, float)) else str(v)
                        for k, v in (best_program.metadata or {}).items()},
        },
        "archive": elites,  # Full archive with all elites and their complete code
    }

    if extra_info:
        state.update(extra_info)

    # Write atomically
    temp_file = STATE_FILE.with_suffix(".json.tmp")
    with open(temp_file, "w") as f:
        json.dump(state, f, indent=2)
    temp_file.rename(STATE_FILE)


def format_metrics_for_llm(
    metrics: dict,
    previous_advice: str,
    best_score: float,
    top_solutions: list[tuple[float, str]] = None,
    progress_pct: float = 0.0,
) -> str:
    """Format metrics data for the LLM to analyze.

    Args:
        metrics: Dict tracking outcomes over recent generations
        previous_advice: Previous meta-advice (for continuity)
        best_score: Current best score achieved
        top_solutions: List of (score, code_snippet) for top solution(s)
        progress_pct: Percentage of budget consumed (0-100)

    Returns:
        Formatted string with all metrics data
    """
    total = metrics.get('total', 0)
    timeouts = metrics.get('timeouts', 0)
    error_count = metrics.get('errors', 0)
    rejections = metrics.get('rejections', 0)
    acceptances = metrics.get('acceptances', 0)
    new_bests = metrics.get('new_bests', 0)
    error_messages = metrics.get('error_messages', set())

    data = f"""## Progress: {progress_pct:.0f}% of budget consumed

## Current Best Score: {best_score:.1f}

## Last 10 Generations ({total} candidates):
- Acceptances: {acceptances} (improved archive)
- Rejections: {rejections} (valid but didn't improve)
- Errors: {error_count} (crashed/invalid)
- Timeouts: {timeouts}
- New Bests: {new_bests}"""

    if error_messages:
        data += "\n\n## Errors Encountered:\n"
        for err in sorted(error_messages):
            data += f"- {err}\n"

    if top_solutions:
        data += f"\n\n## Best Solution (Score: {top_solutions[0][0]:.1f}):\n"
        data += f"```python\n{top_solutions[0][1]}\n```\n"

    if previous_advice:
        data += f"\n\n## Previous Advice:\n{previous_advice}"

    return data


async def generate_meta_advice(
    metrics: dict,
    previous_advice: str,
    best_score: float,
    top_solutions: list[tuple[float, str]] = None,
    model: str = None,
    progress_pct: float = 0.0,
) -> tuple[str, float]:
    """Use LLM to generate strategic meta-advice from evolution metrics.

    The advisor learns from previous advice effectiveness, analyzes error patterns,
    and provides actionable guidance for the next generation of solutions.

    Args:
        metrics: Dict tracking outcomes over recent generations
        previous_advice: Previous meta-advice (advisor should learn from it)
        best_score: Current best score achieved
        top_solutions: List of (score, code_snippet) for top solution(s)
        model: LLM model to use for generating advice
        progress_pct: Percentage of budget consumed (0-100)

    Returns:
        Tuple of (advice string ~500 words, cost)
    """
    metrics_data = format_metrics_for_llm(metrics, previous_advice, best_score, top_solutions, progress_pct)
    prompt = META_ADVISOR_PROMPT.format(metrics_data=metrics_data)

    try:
        # Build call kwargs
        call_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 800,  # ~500 words max
            "timeout": 60,
        }

        # Enable reasoning for DeepSeek models
        if model and "deepseek" in model.lower():
            call_kwargs["reasoning"] = {"enabled": True}

        response = await litellm.acompletion(**call_kwargs)
        advice = response.choices[0].message.content.strip()
        cost = litellm.completion_cost(completion_response=response)
        return advice, cost
    except Exception as e:
        # Fallback to simple formatted advice if LLM fails
        fallback = f"(Meta-advice generation failed: {str(e)[:50]})\n\n"
        fallback += f"Best score: {best_score:.1f}. "
        fallback += f"Last 10 gens: {metrics.get('acceptances', 0)} accepted, {metrics.get('errors', 0)} errors."
        return fallback, 0.0


async def generate_for_island(island_idx: int, prompt: str, model: str, temperature: float) -> dict:
    """Generate code for one island asynchronously."""
    start = time.time()
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=30000,
            timeout=300,
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content
        cost = litellm.completion_cost(completion_response=response)
        tokens = response.usage.total_tokens
        print(f"  [LLM] Island {island_idx} done in {elapsed:.1f}s, {tokens} tokens")
        return {
            "island": island_idx,
            "content": content,
            "cost": cost,
            "tokens": tokens,
            "model": model,
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"  [LLM] Island {island_idx} FAILED after {elapsed:.1f}s: {e}")
        return {"island": island_idx, "error": str(e)}


def main():
    import random

    # Initialize workloads in main process
    global WORKLOADS, BASELINE, EFFECTIVE_OPTIMAL
    WORKLOADS, BASELINE, EFFECTIVE_OPTIMAL, _, _ = _init_workloads()

    # Model configuration: multiple cheap light models for diversity
    LIGHT_MODELS = [
        'openrouter/qwen/qwen3-coder-30b-a3b-instruct',  # $0.07/$0.27 per M
        'openrouter/google/gemini-2.5-flash-lite',       # $0.10/$0.40 per M
        'openrouter/deepseek/deepseek-v3.2',             # $0.26/$0.38 per M
    ]
    HEAVY_MODEL = 'openrouter/deepseek/deepseek-v3.2'

    n_workers = 4
    n_inspirations = 2  # Number of inspiration programs to use alongside parent

    # Use custom behavior extractor with z-score normalization for static features
    # Features: execution_time, loop_count, branch_count, math_operators,
    #           workload_1_score, workload_2_score, workload_3_score
    extractor = TxnSchedulingBehaviorExtractor(max_time=200.0)

    # Single archive with deferred centroid initialization
    pool = CVTMAPElitesPool(
        behavior_extractor=extractor,
        n_centroids=50,  # 50 centroids for more diversity
        subscore_keys=["execution_time"],
        defer_centroids=True,  # Build centroids from initial LLM generations
    )

    # Evaluate only the main seed program
    print(f"Evaluating seed program...", flush=True)

    seed_output = evaluate_code_in_process(SEED_PROGRAM)
    seed_score = score_fn(seed_output)
    seed_exec_time = seed_output.get("wall_time", 60.0)
    best_makespan = seed_output.get("makespan", BASELINE)
    print(f"  Seed: makespan={best_makespan:.0f}, score={seed_score:.1f}, time={seed_exec_time:.1f}s", flush=True)

    seed_program = Program(code=SEED_PROGRAM, metadata={
        "execution_time": seed_exec_time,
        "primary_score": seed_score,
        "workload_1_score": seed_output.get("workload_1_score", 0.0),
        "workload_2_score": seed_output.get("workload_2_score", 0.0),
        "workload_3_score": seed_output.get("workload_3_score", 0.0),
    })
    best_score = seed_score
    best_program = seed_program

    # Get sampler names for display
    sampler_names = pool.get_sampler_names()

    print(f"{'='*70}")
    print(f"AlphaEvolve - Transaction Scheduling (Multi-Sampler)")
    print(f"{'='*70}")
    print(f"  Baseline makespan:    {BASELINE}")
    print(f"  Effective optimal:    {EFFECTIVE_OPTIMAL:.0f}")
    print(f"  Seed makespan:        {best_makespan:.0f} (score: {best_score:.1f})")
    print(f"  Budget:               ${BUDGET_USD:.2f}")
    print(f"  Samplers:             {', '.join(sampler_names)} (UCB runs 3x)")
    print(f"  Light models:         {', '.join(LIGHT_MODELS)}")
    print(f"  Heavy model:          {HEAVY_MODEL} (every gen)")
    print(f"{'='*70}\n")

    generation = 0
    total_cost = 0.0

    # Create resilient process pool with worker recycling to prevent memory buildup
    # Workers restart after 5 tasks, releasing accumulated memory back to OS
    # ResilientProcessPool auto-recovers if workers crash (OOM, segfault, etc.)
    executor = ResilientProcessPool(max_workers=n_workers, max_tasks_per_child=5)

    async def initialize_archive():
        """Generate diverse seeds with heavy model, then expand with light models."""
        nonlocal total_cost, best_score, best_program, best_makespan, seed_score

        loop = asyncio.get_event_loop()

        # Phase 1: Generate 5 diverse seeds using heavy model (sequential, with context)
        n_diverse_seeds = 5
        print(f"\n[Init Phase 1] Generating {n_diverse_seeds} diverse seeds with heavy model...", flush=True)

        # Track seeds with their scores: list of (code, score)
        diverse_seeds = [(SEED_PROGRAM, seed_score)]  # Start with original seed and its score

        for i in range(n_diverse_seeds):
            # Build prompt with all existing seeds and their scores in context
            existing_seeds_text = "\n\n---\n\n".join([
                f"### Seed {j+1} (Score: {score:.1f}):\n```python\n{code}\n```"
                for j, (code, score) in enumerate(diverse_seeds)
            ])
            prompt = DIVERSITY_SEED_PROMPT.format(existing_seeds=existing_seeds_text)

            print(f"  [Seed {i+1}/{n_diverse_seeds}] Generating with {len(diverse_seeds)} seeds in context...", flush=True)

            result = await generate_for_island(i, prompt, 'openrouter/z-ai/glm-4.7', 0.7)

            if "error" in result:
                print(f"  [Seed {i+1}] ERROR: {result['error'][:50]}", flush=True)
                continue

            total_cost += result["cost"]
            new_code, validation_error = extract_and_validate_code(result["content"])

            if new_code:
                # Quick eval to verify it works
                try:
                    eval_result = await loop.run_in_executor(executor.executor, evaluate_code_in_process, new_code)
                except BrokenExecutor:
                    executor._recreate_executor()
                    eval_result = {"error": "Pool crashed, recreated"}
                if "error" not in eval_result:
                    new_score = score_fn(eval_result)
                    diverse_seeds.append((new_code, new_score))
                    print(f"  [Seed {i+1}] OK - score: {new_score:.1f}, tokens: {result['tokens']}", flush=True)
                else:
                    print(f"  [Seed {i+1}] EVAL FAIL: {eval_result['error'][:50]}", flush=True)
            else:
                print(f"  [Seed {i+1}] VALIDATION FAIL: {validation_error}", flush=True)

        print(f"[Init Phase 1] Generated {len(diverse_seeds)-1} new diverse seeds (total: {len(diverse_seeds)})", flush=True)

        # Phase 2: Generate 150 variants using light models (25 per diverse seed)
        import random
        n_variants_per_seed = 25
        n_variants = n_variants_per_seed * len(diverse_seeds)
        print(f"\n[Init Phase 2] Generating {n_variants} variants ({n_variants_per_seed} per seed) with light models...", flush=True)

        # Build prompts: 25 variants for each diverse seed
        prompts = []
        prompt_seed_idx = []  # Track which seed each prompt is for
        for seed_idx, (seed_code, s_score) in enumerate(diverse_seeds):
            seed_prog = Program(code=seed_code, metadata={"score": s_score})
            # Create EvaluationResult for ProgramWithScore
            seed_eval_result = EvaluationResult(
                program_id=seed_prog.id,
                scores={'score': s_score},
                is_valid=True,
            )
            for _ in range(n_variants_per_seed):
                builder = PromptBuilder()
                builder.add_section("Problem", PROBLEM_DESCRIPTION, priority=10)
                builder.add_section("Signature", f"```python\n{FUNCTION_SIGNATURE}\n```", priority=20)
                builder.add_parents([ProgramWithScore(seed_prog, seed_eval_result)], priority=30)
                builder.set_output_mode(OutputMode.FULL)
                prompts.append(builder.build())
                prompt_seed_idx.append(seed_idx)

        # Make parallel LLM calls (use flash lite for init)
        llm_tasks = [
            generate_for_island(i, prompts[i], 'openrouter/google/gemini-2.5-flash-lite', 0.8)
            for i in range(n_variants)
        ]
        results = await asyncio.gather(*llm_tasks)
        print(f"[Init Phase 2] All {n_variants} LLM calls complete", flush=True)

        # Extract code from responses
        candidates = []
        for res in results:
            if "error" in res:
                print(f"[Init] Candidate {res['island']} ERROR: {res['error'][:50]}", flush=True)
                continue

            total_cost += res["cost"]
            idx = res["island"]
            tokens = res.get("tokens", 0)
            model = res.get("model", "unknown")

            new_code, validation_error = extract_and_validate_code(res["content"])

            if not new_code:
                n_extract_fail = getattr(initialize_archive, 'n_extract_fail', 0) + 1
                initialize_archive.n_extract_fail = n_extract_fail
                if n_extract_fail <= 5:  # Show first 5 validation errors
                    print(f"[Init] Candidate {idx} VALIDATION FAIL ({model.split('/')[-1]}): {validation_error}", flush=True)
                continue

            candidates.append({"idx": idx, "code": new_code, "tokens": tokens, "model": model})

        print(f"[Init] Evaluating {len(candidates)} candidates...", flush=True)

        # Evaluate with semaphore to ensure timeout starts when task actually runs
        eval_map = {}
        completed = 0
        semaphore = asyncio.Semaphore(4)  # Match executor workers

        async def eval_candidate(idx, code):
            nonlocal completed
            async with semaphore:  # Only start timeout when we acquire semaphore
                start = time.time()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(executor.executor, evaluate_code_in_process, code),
                        timeout=60  # 1 minute timeout for init phase
                    )
                    elapsed = time.time() - start
                    completed += 1
                    print(f"  [Init Eval] {completed}/{len(candidates)} Candidate {idx} done in {elapsed:.1f}s", flush=True)
                    return idx, result
                except asyncio.TimeoutError:
                    completed += 1
                    # Recreate executor to kill stuck worker (same issue as main eval loop)
                    executor._recreate_executor()
                    print(f"  [Init Eval] {completed}/{len(candidates)} Candidate {idx} TIMEOUT (pool recreated)", flush=True)
                    return idx, {"error": "Timeout"}
                except BrokenExecutor as e:
                    # Pool crashed - recreate it and return error for this candidate
                    executor._recreate_executor()
                    completed += 1
                    print(f"  [Init Eval] {completed}/{len(candidates)} Candidate {idx} POOL CRASHED (recreated)", flush=True)
                    return idx, {"error": "Pool crashed, recreated"}
                except Exception as e:
                    completed += 1
                    print(f"  [Init Eval] {completed}/{len(candidates)} Candidate {idx} ERROR: {e}", flush=True)
                    return idx, {"error": str(e)}

        eval_tasks = [eval_candidate(c["idx"], c["code"]) for c in candidates]
        eval_results = await asyncio.gather(*eval_tasks)
        eval_map = {idx: res for idx, res in eval_results}

        print(f"[Init] Processing {len(candidates)} candidates...", flush=True)

        # Collect valid programs and their behavior vectors
        valid_programs = []
        behavior_vectors = []
        n_errors = 0
        for cand in candidates:
            output = eval_map.get(cand["idx"], {"error": "missing"})
            if "error" in output:
                n_errors += 1
                if n_errors <= 5:
                    print(f"[Init] Candidate {cand['idx']} EVAL FAIL: {output['error'][:80]}", flush=True)
                continue
            score = score_fn(output)
            execution_time = output.get("wall_time", 60.0)
            valid_programs.append({
                "code": cand["code"],
                "score": score,
                "output": output,
                "execution_time": execution_time,
            })
            temp_prog = Program(code=cand["code"], metadata={
                "execution_time": execution_time,
                "primary_score": score,
                "workload_1_score": output.get("workload_1_score", 0.0),
                "workload_2_score": output.get("workload_2_score", 0.0),
                "workload_3_score": output.get("workload_3_score", 0.0),
            })
            behavior = extractor.extract(temp_prog)
            behavior_vectors.append(np.array([behavior[f] for f in extractor.features]))

        # Print total errors if more than 5
        if n_errors > 5:
            print(f"[Init] ... and {n_errors - 5} more eval failures (total: {n_errors})", flush=True)

        print(f"[Init] Valid programs: {len(valid_programs)}/{len(candidates)} ({n_errors} eval failures)", flush=True)

        # Add all diverse seeds (including original) to valid_programs
        print(f"[Init] Adding {len(diverse_seeds)} diverse seeds to candidates...", flush=True)
        for seed_code, seed_score in diverse_seeds:
            try:
                seed_eval = await loop.run_in_executor(executor.executor, evaluate_code_in_process, seed_code)
            except BrokenExecutor:
                executor._recreate_executor()
                seed_eval = {"error": "Pool crashed, recreated"}
            if "error" not in seed_eval:
                execution_time = seed_eval.get("wall_time", 60.0)
                valid_programs.append({
                    "code": seed_code,
                    "score": seed_score,  # Use already computed score
                    "output": seed_eval,
                    "execution_time": execution_time,
                })
                temp_prog = Program(code=seed_code, metadata={
                    "execution_time": execution_time,
                    "primary_score": seed_score,
                    "workload_1_score": seed_eval.get("workload_1_score", 0.0),
                    "workload_2_score": seed_eval.get("workload_2_score", 0.0),
                    "workload_3_score": seed_eval.get("workload_3_score", 0.0),
                })
                behavior = extractor.extract(temp_prog)
                behavior_vectors.append(np.array([behavior[f] for f in extractor.features]))

        print(f"[Init] Total valid programs: {len(valid_programs)}", flush=True)

        # Build centroids from ALL valid programs' behavior vectors (for better CVT coverage)
        print(f"[Init] Building centroids from {len(behavior_vectors)} behavior vectors (all surviving programs)...", flush=True)
        n_centroids = pool.set_centroids_from_data(
            behavior_vectors,
            percentile_low=5.0,
            percentile_high=95.0,
            n_centroids=50,
        )
        print(f"[Init] Built {n_centroids} centroids", flush=True)

        # Select top 50 scoring programs to populate the archive
        valid_programs.sort(key=lambda x: x["score"], reverse=True)
        top_programs = valid_programs[:50]
        print(f"[Init] Selected top {len(top_programs)} programs to populate archive", flush=True)

        # Add top programs to archive
        n_accepted = 0
        for prog in top_programs:
            output = prog.get("output", {})
            child = Program(code=prog["code"], metadata={
                "execution_time": prog["execution_time"],
                "primary_score": prog["score"],
                "workload_1_score": output.get("workload_1_score", 0.0),
                "workload_2_score": output.get("workload_2_score", 0.0),
                "workload_3_score": output.get("workload_3_score", 0.0),
            })
            eval_result = EvaluationResult(
                program_id=child.id,
                scores={'score': prog["score"]},
                is_valid=True,
            )
            if pool.add(child, eval_result):
                n_accepted += 1
                if prog["score"] > best_score:
                    best_score = prog["score"]
                    best_program = child
                    best_makespan = prog["output"].get("makespan", BASELINE)

        print(f"[Init] Done: {n_accepted} accepted, archive size: {pool.size()}, best: {best_score:.1f}, cost: ${total_cost:.3f}\n", flush=True)

        # Save initial state
        save_state(
            generation=0,
            pool=pool,
            best_score=best_score,
            best_program=best_program,
            total_cost=total_cost,
            extra_info={"phase": "initialization", "candidates_accepted": n_accepted},
        )
        print(f"[Init] State saved to {STATE_FILE}\n", flush=True)

    async def run_pipeline():
        """
        Producer-consumer pipeline for async LLM sampling and evaluation.

        Architecture:
        - Sampler Queue: Yields (display_name, real_sampler, model) in fair distribution
        - LLM Producers: Pull from sampler queue, generate code, push to eval queue
        - Eval Consumers: Pull from eval queue, evaluate in process pool, push to result queue
        - Result Processor: Update archive, track metrics, trigger meta-advice every 50 evals
        """
        nonlocal generation, best_score, best_program, best_makespan, total_cost

        print(f"[Pipeline] Starting async evolution pipeline...", flush=True)
        loop = asyncio.get_event_loop()

        # Configuration for 8-core machine
        # LLM: I/O bound (10-30s) - just need enough async tasks to keep eval fed
        # Eval: CPU bound (1-300s) - need 8 processes for 8 cores
        #
        # Math: At avg 60s eval time, 8 workers process ~8 evals/min
        #       At avg 20s LLM time, need ~3 concurrent LLM calls to match
        #       Use 8 for buffer (they're just async, no CPU cost)
        N_LLM_WORKERS = 4       # Concurrent async LLM calls (I/O bound)
        N_EVAL_PROCESSES = 4    # Process pool size (CPU bound, matches cores)
        EVAL_TIMEOUT = 300      # 5 minute timeout
        META_ADVICE_INTERVAL = 50  # Generate meta-advice every N evals

        # Queues for pipeline
        sampler_queue = asyncio.Queue()      # Sampler configs to process
        eval_queue = asyncio.Queue()         # Candidates awaiting evaluation (unbounded, gated by semaphore)
        result_queue = asyncio.Queue()       # Evaluated results

        # Semaphore to limit total in-flight work (eval running + queued + LLM generating)
        # This blocks BEFORE LLM call, not after, so we don't waste money
        # 8 eval processes + 4 buffer = 12 max concurrent candidates
        pipeline_capacity = asyncio.Semaphore(N_EVAL_PROCESSES + 4)

        # Shared state with lock
        state_lock = asyncio.Lock()
        state = {
            'total_cost': total_cost,
            'eval_count': 0,
            'best_score': best_score,
            'best_program': best_program,
            'best_makespan': best_makespan,
            'current_meta_advice': '',
            'previous_meta_advice': '',
            'period_metrics': {
                'total': 0,
                'timeouts': 0,
                'errors': 0,
                'rejections': 0,
                'acceptances': 0,
                'new_bests': 0,
                'error_messages': set(),
            },
            'llm_in_flight': 0,
            'eval_in_flight': 0,
            'consecutive_timeouts': 0,  # Track for pool recreation
        }

        # Stop signal
        stop_event = asyncio.Event()

        def get_sampler_cycle():
            """Generate one cycle of sampler configs with fair distribution."""
            import random as rand
            configs = []
            for name in sampler_names:
                sampler = pool.get_sampler(name)
                if sampler.model_type == "heavy":
                    configs.append((name, name, HEAVY_MODEL))
                elif name == "ucb":
                    # 3x UCB with different light models
                    for ucb_idx in range(3):
                        configs.append((f"ucb_{ucb_idx}", "ucb", rand.choice(LIGHT_MODELS)))
                else:
                    configs.append((name, name, rand.choice(LIGHT_MODELS)))
            return configs

        async def sampler_feeder():
            """Continuously feed sampler configs to the queue."""
            cycle_num = 0
            while not stop_event.is_set():
                cycle_num += 1
                configs = get_sampler_cycle()
                for config in configs:
                    if stop_event.is_set():
                        break
                    await sampler_queue.put(config)
                # Small delay between cycles to allow checking stop
                await asyncio.sleep(0.1)

        async def llm_producer(worker_id: int):
            """Pull sampler config, sample from pool, generate LLM response, push to eval queue."""
            import random as rand
            while not stop_event.is_set():
                slot_acquired = False
                slot_transferred = False
                try:
                    # Get next sampler config (with timeout to check stop)
                    try:
                        display_name, real_sampler, model = await asyncio.wait_for(
                            sampler_queue.get(), timeout=2.0
                        )
                    except asyncio.TimeoutError:
                        continue

                    # Acquire pipeline slot BEFORE LLM call to avoid wasted API costs
                    # This blocks if too many candidates are in-flight (eval + queued)
                    await pipeline_capacity.acquire()
                    slot_acquired = True

                    # Check budget before starting
                    async with state_lock:
                        if state['total_cost'] >= BUDGET_USD:
                            stop_event.set()
                            break  # slot released in finally
                        state['llm_in_flight'] += 1
                        current_meta_advice = state['current_meta_advice']

                    # Sample from pool (thread-safe read)
                    sample = pool.sample(real_sampler, n_parents=1 + n_inspirations)
                    inspirations = [p for p in sample.inspirations if rand.random() < 0.8]
                    parents = [sample.parent] + inspirations
                    source_cell = sample.metadata.get("source_cell", 0)

                    # Build prompt
                    builder = PromptBuilder()
                    builder.add_section("Problem", PROBLEM_DESCRIPTION, priority=10)
                    builder.add_section("Signature", f"```python\n{FUNCTION_SIGNATURE}\n```", priority=20)
                    builder.add_parents([ProgramWithScore(p, None) for p in parents], priority=30)
                    builder.set_output_mode(OutputMode.FULL)

                    if current_meta_advice and rand.random() < 0.8:
                        builder.add_section("Meta-Advice", current_meta_advice, priority=100)

                    prompt = builder.build()

                    # Generate
                    result = await generate_for_island(worker_id, prompt, model, 0.8)

                    async with state_lock:
                        state['llm_in_flight'] -= 1

                    if "error" in result:
                        print(f"[LLM-{worker_id}] {display_name} ERROR: {result['error'][:50]}", flush=True)
                        continue  # slot released in finally

                    # Track cost
                    async with state_lock:
                        state['total_cost'] += result["cost"]
                        if state['total_cost'] >= BUDGET_USD:
                            stop_event.set()

                    # Extract code
                    new_code, validation_error = extract_and_validate_code(result["content"])
                    if not new_code:
                        print(f"[LLM-{worker_id}] {display_name} VALIDATION FAIL: {validation_error}", flush=True)
                        continue  # slot released in finally

                    # Push to eval queue (slot transfers to eval, released after eval completes)
                    # Use shield to ensure put completes even if cancelled during transfer
                    # This prevents race condition where put succeeds but slot_transferred not set
                    candidate = {
                        "sampler": display_name,
                        "real_sampler": real_sampler,
                        "code": new_code,
                        "tokens": result.get("tokens", 0),
                        "source_cell": source_cell,
                        "model": model,
                    }
                    try:
                        await asyncio.shield(eval_queue.put(candidate))
                        slot_transferred = True  # Eval now owns the slot
                    except asyncio.CancelledError:
                        # Put completed (due to shield) but we're being cancelled
                        # Mark slot as transferred before re-raising
                        slot_transferred = True
                        async with state_lock:
                            if state['llm_in_flight'] > 0:
                                state['llm_in_flight'] -= 1
                        raise

                except asyncio.CancelledError:
                    # Handle cancellation gracefully - don't log as error
                    async with state_lock:
                        if state['llm_in_flight'] > 0:
                            state['llm_in_flight'] -= 1
                    raise  # Re-raise to properly cancel the task
                except Exception as e:
                    print(f"[LLM-{worker_id}] Unexpected error: {e}", flush=True)
                    async with state_lock:
                        if state['llm_in_flight'] > 0:
                            state['llm_in_flight'] -= 1
                finally:
                    # Release slot if we acquired it but didn't transfer to eval
                    if slot_acquired and not slot_transferred:
                        pipeline_capacity.release()

        async def eval_dispatcher():
            """
            Single dispatcher that pulls from eval queue and spawns eval tasks.
            Uses semaphore to limit concurrent process pool submissions to N_EVAL_PROCESSES.
            """
            eval_semaphore = asyncio.Semaphore(N_EVAL_PROCESSES)
            pending_evals = set()  # Track in-flight eval tasks

            async def run_eval(candidate):
                """Run single evaluation with semaphore-limited concurrency."""
                output = None
                elapsed = 0
                try:
                    async with eval_semaphore:
                        async with state_lock:
                            state['eval_in_flight'] += 1

                        start = time.time()
                        try:
                            output = await asyncio.wait_for(
                                loop.run_in_executor(executor.executor, evaluate_code_in_process, candidate["code"]),
                                timeout=EVAL_TIMEOUT
                            )
                            elapsed = time.time() - start
                        except asyncio.TimeoutError:
                            output = {"error": f"Timeout after {EVAL_TIMEOUT}s"}
                            elapsed = EVAL_TIMEOUT
                            # Track consecutive timeouts and recreate pool if too many
                            async with state_lock:
                                state['consecutive_timeouts'] += 1
                                if state['consecutive_timeouts'] >= 3:
                                    print(f"  [Pool] {state['consecutive_timeouts']} consecutive timeouts - recreating executor to kill stuck workers", flush=True)
                                    executor._recreate_executor()
                                    state['consecutive_timeouts'] = 0
                        except BrokenExecutor:
                            executor._recreate_executor()
                            output = {"error": "Pool crashed, recreated"}
                            elapsed = time.time() - start
                        except asyncio.CancelledError:
                            output = {"error": "Evaluation cancelled"}
                            elapsed = time.time() - start
                            raise  # Re-raise to propagate cancellation
                        except Exception as e:
                            output = {"error": str(e)}
                            elapsed = time.time() - start
                        finally:
                            # Always release pipeline slot and update state
                            async with state_lock:
                                state['eval_in_flight'] -= 1
                            pipeline_capacity.release()

                        await result_queue.put({
                            "candidate": candidate,
                            "output": output,
                            "elapsed": elapsed,
                        })
                except asyncio.CancelledError:
                    # If cancelled, still try to push result if we have one
                    if output is not None:
                        try:
                            await asyncio.shield(result_queue.put({
                                "candidate": candidate,
                                "output": output,
                                "elapsed": elapsed,
                            }))
                        except Exception:
                            pass
                    raise
                except Exception as e:
                    # Log unexpected errors that would otherwise be silently lost
                    print(f"[Eval] Unexpected error in run_eval: {e}", flush=True)

            while not stop_event.is_set() or not eval_queue.empty():
                try:
                    candidate = await asyncio.wait_for(eval_queue.get(), timeout=2.0)
                    # Spawn eval task (semaphore limits actual concurrency)
                    task = asyncio.create_task(run_eval(candidate))
                    pending_evals.add(task)
                    task.add_done_callback(pending_evals.discard)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    # If cancelled, break the loop and wait for pending evals
                    break
                except Exception as e:
                    print(f"[EvalDispatch] Unexpected error: {e}", flush=True)

            # Wait for all pending evals to complete (with timeout to avoid hanging)
            if pending_evals:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending_evals, return_exceptions=True),
                        timeout=EVAL_TIMEOUT + 10  # Give extra time beyond individual eval timeout
                    )
                except asyncio.TimeoutError:
                    print(f"[EvalDispatch] Pending evals didn't complete in time, continuing...", flush=True)

        async def result_processor():
            """Process evaluation results, update archive, trigger meta-advice."""
            nonlocal generation, best_score, best_program, best_makespan, total_cost

            last_save_time = time.time()

            while not stop_event.is_set() or not result_queue.empty():
                try:
                    # Get next result
                    try:
                        item = await asyncio.wait_for(result_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    candidate = item["candidate"]
                    output = item["output"]
                    elapsed = item["elapsed"]

                    display_name = candidate["sampler"]
                    real_sampler = candidate["real_sampler"]
                    tokens = candidate["tokens"]

                    score = score_fn(output)
                    exec_time = output.get("wall_time", 60.0)

                    child = Program(code=candidate["code"], metadata={
                        "execution_time": exec_time,
                        "primary_score": score,
                        "workload_1_score": output.get("workload_1_score", 0.0),
                        "workload_2_score": output.get("workload_2_score", 0.0),
                        "workload_3_score": output.get("workload_3_score", 0.0),
                    })
                    eval_result = EvaluationResult(
                        program_id=child.id,
                        scores={'score': score},
                        is_valid="error" not in output,
                        error=output.get("error"),
                    )

                    async with state_lock:
                        state['eval_count'] += 1
                        state['period_metrics']['total'] += 1
                        eval_count = state['eval_count']
                        current_cost = state['total_cost']

                    if eval_result.is_valid:
                        accepted = pool.add(child, eval_result)
                        pool.update_sampler(real_sampler, candidate["source_cell"], success=accepted, reward=score)
                        makespan = output.get("makespan", BASELINE)
                        wall_time = exec_time * 1000

                        async with state_lock:
                            # Reset timeout counter on successful eval
                            state['consecutive_timeouts'] = 0
                            if accepted:
                                state['period_metrics']['acceptances'] += 1
                            else:
                                state['period_metrics']['rejections'] += 1

                            status = "accepted" if accepted else "rejected"
                            if score > state['best_score']:
                                state['best_score'] = score
                                state['best_program'] = child
                                state['best_makespan'] = makespan
                                best_score = score
                                best_program = child
                                best_makespan = makespan
                                status = "NEW BEST ★"
                                state['period_metrics']['new_bests'] += 1

                        print(f"[Eval #{eval_count:4d}] {display_name:20s} {status:10s} | mkspan: {makespan:5.0f} | score: {score:5.1f} | time: {wall_time:6.0f}ms | best: {best_score:5.1f} | {tokens}tok | ${current_cost:.3f}", flush=True)
                    else:
                        pool.update_sampler(real_sampler, candidate["source_cell"], success=False)
                        err = eval_result.error[:30] if eval_result.error else "unknown"

                        async with state_lock:
                            if eval_result.error and "timeout" in eval_result.error.lower():
                                state['period_metrics']['timeouts'] += 1
                            else:
                                state['period_metrics']['errors'] += 1
                                if eval_result.error:
                                    state['period_metrics']['error_messages'].add(eval_result.error[:100].strip())

                        print(f"[Eval #{eval_count:4d}] {display_name:20s} INVALID    | {err}...", flush=True)

                    # Trigger meta-advice every N evals
                    if eval_count > 0 and eval_count % META_ADVICE_INTERVAL == 0:
                        asyncio.create_task(generate_and_update_meta_advice(eval_count))

                    # Save state periodically (every 30 seconds)
                    if time.time() - last_save_time > 30:
                        async with state_lock:
                            save_state(
                                generation=eval_count,
                                pool=pool,
                                best_score=state['best_score'],
                                best_program=state['best_program'],
                                total_cost=state['total_cost'],
                                extra_info={
                                    "phase": "pipeline",
                                    "eval_count": eval_count,
                                    "llm_in_flight": state['llm_in_flight'],
                                    "eval_in_flight": state['eval_in_flight'],
                                },
                            )
                        last_save_time = time.time()

                except asyncio.CancelledError:
                    # If cancelled, do final save and re-raise
                    print(f"[ResultProc] Cancelled, saving final state...", flush=True)
                    break
                except Exception as e:
                    print(f"[ResultProc] Unexpected error: {e}", flush=True)

            # Final save
            async with state_lock:
                total_cost = state['total_cost']
                best_score = state['best_score']
                best_program = state['best_program']
                best_makespan = state['best_makespan']
                generation = state['eval_count']

        async def generate_and_update_meta_advice(eval_count: int):
            """Generate meta-advice asynchronously and update shared state."""
            try:
                async with state_lock:
                    metrics_copy = dict(state['period_metrics'])
                    # Create independent copy of the set to avoid race conditions
                    metrics_copy['error_messages'] = set(state['period_metrics']['error_messages'])
                    prev_advice = state['previous_meta_advice']
                    current_best = state['best_score']
                    current_cost = state['total_cost']

                # Get top 1 solution from archive (reduced from 3 to save tokens)
                top_solutions = []
                try:
                    elites = list(pool._elites.values())
                    elites.sort(key=lambda e: e.result.primary_score, reverse=True)
                    for elite in elites[:1]:
                        top_solutions.append((elite.result.primary_score, elite.program.code))
                except Exception:
                    pass

                progress_pct = (current_cost / BUDGET_USD) * 100
                print(f"\n[Meta-Advice] Generating at eval #{eval_count} ({progress_pct:.0f}% budget)...", flush=True)

                advice, advice_cost = await generate_meta_advice(
                    metrics=metrics_copy,
                    previous_advice=prev_advice,
                    best_score=current_best,
                    top_solutions=top_solutions,
                    model=HEAVY_MODEL,
                    progress_pct=progress_pct,
                )

                async with state_lock:
                    state['total_cost'] += advice_cost
                    state['previous_meta_advice'] = advice
                    state['current_meta_advice'] = advice
                    # Reset period metrics
                    state['period_metrics'] = {
                        'total': 0,
                        'timeouts': 0,
                        'errors': 0,
                        'rejections': 0,
                        'acceptances': 0,
                        'new_bests': 0,
                        'error_messages': set(),
                    }

                print(f"[Meta-Advice] Updated (cost: ${advice_cost:.4f})", flush=True)
                print(f"[Meta-Advice]\n{advice}\n", flush=True)
            except asyncio.CancelledError:
                print(f"[Meta-Advice] Cancelled at eval #{eval_count}", flush=True)
                raise
            except Exception as e:
                # Log unexpected errors that would otherwise be silently lost
                print(f"[Meta-Advice] Unexpected error at eval #{eval_count}: {e}", flush=True)

        async def status_monitor():
            """Periodically print pipeline status."""
            while not stop_event.is_set():
                await asyncio.sleep(30)
                async with state_lock:
                    print(f"\n[Status] Cost: ${state['total_cost']:.3f}/{BUDGET_USD:.2f} | Evals: {state['eval_count']} | "
                          f"LLM in-flight: {state['llm_in_flight']} | Eval in-flight: {state['eval_in_flight']} | "
                          f"Archive: {pool.size()} | Best: {state['best_score']:.1f}\n", flush=True)

        # Start all workers
        print(f"[Pipeline] Starting {N_LLM_WORKERS} async LLM workers, {N_EVAL_PROCESSES}-process eval pool...", flush=True)

        feeder_task = asyncio.create_task(sampler_feeder())
        llm_tasks = [asyncio.create_task(llm_producer(i)) for i in range(N_LLM_WORKERS)]
        eval_task = asyncio.create_task(eval_dispatcher())  # Single dispatcher with semaphore
        processor_task = asyncio.create_task(result_processor())
        monitor_task = asyncio.create_task(status_monitor())

        # Wait for budget exhaustion or stop
        while not stop_event.is_set():
            await asyncio.sleep(1.0)
            async with state_lock:
                if state['total_cost'] >= BUDGET_USD:
                    stop_event.set()

        print(f"\n[Pipeline] Budget exhausted, draining queues...", flush=True)

        # Cancel feeder first to stop new work
        feeder_task.cancel()
        try:
            await feeder_task
        except asyncio.CancelledError:
            pass

        # Cancel LLM workers with timeout (they might be stuck on API calls)
        for task in llm_tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*llm_tasks, return_exceptions=True),
                timeout=10.0  # Give 10s for LLM calls to cancel gracefully
            )
        except asyncio.TimeoutError:
            print("[Pipeline] LLM workers didn't cancel in time, continuing...", flush=True)

        # Wait for eval dispatcher to drain (it waits for pending evals internally)
        await eval_task

        # Wait for result processor to naturally drain (stop_event is set, it will exit
        # when result_queue is empty). Give it reasonable time to finish.
        try:
            await asyncio.wait_for(processor_task, timeout=30.0)
        except asyncio.TimeoutError:
            print("[Pipeline] Result processor didn't finish in time, cancelling...", flush=True)
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass

        # Cancel monitor
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        # Final state update
        async with state_lock:
            total_cost = state['total_cost']
            best_score = state['best_score']
            best_program = state['best_program']
            best_makespan = state['best_makespan']
            generation = state['eval_count']

        # Final save
        save_state(
            generation=generation,
            pool=pool,
            best_score=best_score,
            best_program=best_program,
            total_cost=total_cost,
            extra_info={"phase": "pipeline_complete"},
        )

        executor.shutdown(wait=False)
        print(f"[Pipeline] Complete. Total evals: {generation}", flush=True)

    async def main():
        await initialize_archive()
        await run_pipeline()

    asyncio.run(main())

    print(f"\n{'='*70}")
    print(f"Complete | Generations: {generation}")
    print(f"Best makespan: {best_makespan:.0f} | Score: {best_score:.1f}")
    print(f"{'='*70}\n")

    out = Path(__file__).parent / "best_solution.py"
    out.write_text(best_program.code)
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
