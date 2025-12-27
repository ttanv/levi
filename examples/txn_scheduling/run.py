#!/usr/bin/env python3
"""
AlphaEvolve on Transaction Scheduling Problem.
"""

import asyncio
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Add algoforge to path
ALGOFORGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ALGOFORGE_ROOT))

# Add txn_scheduling resources to path
TXN_RESOURCES = ALGOFORGE_ROOT.parent / "ADRS-Leaderboard" / "problems" / "txn_scheduling" / "resources"
sys.path.insert(0, str(TXN_RESOURCES))

from algoforge.core import Program, EvaluationResult
from algoforge.pool import CVTMAPElitesPool
from algoforge.llm import PromptBuilder, ProgramWithScore, OutputMode
from algoforge.behavior import GeneralizedBehaviorExtractor
import re
import json
import numpy as np

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


def _init_workloads():
    """Initialize workloads (called once per process)."""
    global _WORKLOADS, _BASELINE, _EFFECTIVE_OPTIMAL
    if _WORKLOADS is None:
        _WORKLOADS = [Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)]
        _BASELINE = sum(w.get_opt_seq_cost(list(range(w.num_txns))) for w in _WORKLOADS)
        theoretical_optimal = sum(max(txn[0][3] for txn in w.txns) for w in _WORKLOADS)
        _EFFECTIVE_OPTIMAL = theoretical_optimal + 0.10 * (_BASELINE - theoretical_optimal)
    return _WORKLOADS, _BASELINE, _EFFECTIVE_OPTIMAL


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

    workloads, baseline, _ = _init_workloads()

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
        makespan, schedules, algo_time = namespace['get_random_costs']()
        wall_time = time_module.time() - eval_start

        # Get constraint stats
        constraint_stats = ConstraintEnforcingWorkload.get_stats()

        # Validate schedules
        for i, sched in enumerate(schedules):
            if set(sched) != set(range(workloads[i].num_txns)):
                return {"error": f"Invalid schedule for workload {i}"}

        return {
            "makespan": makespan,
            "algo_time": algo_time,
            "wall_time": wall_time,
            "get_opt_seq_cost_calls": constraint_stats["total_calls"],
            "max_allowed_calls": constraint_stats["max_allowed"],
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
) -> str:
    """Format metrics data for the LLM to analyze.

    Args:
        metrics: Dict tracking outcomes over recent generations
        previous_advice: Previous meta-advice (for continuity)
        best_score: Current best score achieved
        top_solutions: List of (score, code_snippet) for top 3 solutions

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

    data = f"""Current Best Score: {best_score:.1f}

Outcomes from Last 10 Generations ({total} candidates):
- Timeouts: {timeouts} (solution took too long to execute)
- Errors: {error_count} (solution crashed, had bugs, or failed validation)
- Rejections: {rejections} (solution was valid but didn't beat existing archive entry)
- Acceptances: {acceptances} (solution was good enough to enter the archive)
- New Bests: {new_bests} (solution achieved a new highest score)"""

    if error_messages:
        data += "\n\nErrors encountered:\n"
        for err in sorted(error_messages):
            data += f"- {err}\n"

    if top_solutions:
        data += f"\n\nTop {len(top_solutions)} Solutions in Archive:\n"
        for i, (score, snippet) in enumerate(top_solutions, 1):
            data += f"\n### #{i} (Score: {score:.1f})\n```python\n{snippet}\n```\n"

    if previous_advice:
        data += f"\n\nPrevious Strategic Advice:\n{previous_advice}"

    return data


async def generate_meta_advice(
    metrics: dict,
    previous_advice: str,
    best_score: float,
    top_solutions: list[tuple[float, str]] = None,
    model: str = None,
) -> tuple[str, float]:
    """Use heavy LLM to generate strategic meta-advice from evolution metrics.

    Args:
        metrics: Dict tracking outcomes over recent generations
        previous_advice: Previous meta-advice (for continuity)
        best_score: Current best score achieved
        top_solutions: List of (score, code_snippet) for top 3 solutions
        model: LLM model to use for generating advice

    Returns:
        Tuple of (advice string, cost)
    """
    metrics_data = format_metrics_for_llm(metrics, previous_advice, best_score, top_solutions)
    prompt = META_ADVISOR_PROMPT.format(metrics_data=metrics_data)

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,  # ~300 words max
            timeout=60,
        )
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
    WORKLOADS, BASELINE, EFFECTIVE_OPTIMAL = _init_workloads()

    # Model configuration: multiple cheap light models for diversity
    LIGHT_MODELS = [
        'openrouter/qwen/qwen3-coder-30b-a3b-instruct',  # $0.07/$0.27 per M
        'openrouter/google/gemini-2.5-flash-lite',       # $0.10/$0.40 per M
        'openrouter/deepseek/deepseek-v3.2',             # $0.26/$0.38 per M
    ]
    HEAVY_MODEL = 'openrouter/deepseek/deepseek-v3.2'

    n_workers = 8
    n_inspirations = 2  # Number of inspiration programs to use alongside parent

    # Use generalized behavior extractor with standard features
    extractor = GeneralizedBehaviorExtractor(
        feature_set='standard',
        time_key='execution_time',
        score_key='primary_score',
        max_time=200.0,  # 5 minute max
        max_score=100.0,
    )

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
    print(f"  Budget:               $5.00")
    print(f"  Samplers:             {', '.join(sampler_names)} (UCB runs 3x)")
    print(f"  Light models:         {', '.join(LIGHT_MODELS)}")
    print(f"  Heavy model:          {HEAVY_MODEL} (every gen)")
    print(f"{'='*70}\n")

    generation = 0
    total_cost = 0.0

    # Create process pool with worker recycling to prevent memory buildup
    # Workers restart after 5 tasks, releasing accumulated memory back to OS
    executor = ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=5)

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
                eval_result = await loop.run_in_executor(executor, evaluate_code_in_process, new_code)
                if "error" not in eval_result:
                    new_score = score_fn(eval_result)
                    diverse_seeds.append((new_code, new_score))
                    print(f"  [Seed {i+1}] OK - score: {new_score:.1f}, tokens: {result['tokens']}", flush=True)
                else:
                    print(f"  [Seed {i+1}] EVAL FAIL: {eval_result['error'][:50]}", flush=True)
            else:
                print(f"  [Seed {i+1}] VALIDATION FAIL: {validation_error}", flush=True)

        print(f"[Init Phase 1] Generated {len(diverse_seeds)-1} new diverse seeds (total: {len(diverse_seeds)})", flush=True)

        # Phase 2: Generate 100 variants using light models (20 per diverse seed)
        import random
        n_variants_per_seed = 20
        n_variants = n_variants_per_seed * len(diverse_seeds)
        print(f"\n[Init Phase 2] Generating {n_variants} variants ({n_variants_per_seed} per seed) with light models...", flush=True)

        # Build prompts: 20 variants for each diverse seed
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

        # Make parallel LLM calls (alternate between light models)
        llm_tasks = [
            generate_for_island(i, prompts[i], LIGHT_MODELS[i % len(LIGHT_MODELS)], 0.8)
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
        semaphore = asyncio.Semaphore(8)  # Match executor workers

        async def eval_candidate(idx, code):
            nonlocal completed
            async with semaphore:  # Only start timeout when we acquire semaphore
                start = time.time()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(executor, evaluate_code_in_process, code),
                        timeout=60  # 1 minute timeout for init phase
                    )
                    elapsed = time.time() - start
                    completed += 1
                    print(f"  [Init Eval] {completed}/{len(candidates)} Candidate {idx} done in {elapsed:.1f}s", flush=True)
                    return idx, result
                except asyncio.TimeoutError:
                    completed += 1
                    print(f"  [Init Eval] {completed}/{len(candidates)} Candidate {idx} TIMEOUT", flush=True)
                    return idx, {"error": "Timeout"}
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
            seed_eval = await loop.run_in_executor(executor, evaluate_code_in_process, seed_code)
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
                })
                behavior = extractor.extract(temp_prog)
                behavior_vectors.append(np.array([behavior[f] for f in extractor.features]))

        print(f"[Init] Total valid programs: {len(valid_programs)}", flush=True)

        # Select top 50 scoring programs
        valid_programs.sort(key=lambda x: x["score"], reverse=True)
        top_programs = valid_programs[:50]
        print(f"[Init] Selected top {len(top_programs)} programs by score", flush=True)

        # Rebuild behavior vectors for top programs
        top_behaviors = []
        for prog in top_programs:
            temp_prog = Program(code=prog["code"], metadata={
                "execution_time": prog["execution_time"],
                "primary_score": prog["score"],
            })
            behavior = extractor.extract(temp_prog)
            top_behaviors.append(np.array([behavior[f] for f in extractor.features]))

        # Build centroids from top programs
        print(f"[Init] Building centroids from {len(top_behaviors)} behavior vectors...", flush=True)
        n_centroids = pool.set_centroids_from_data(
            top_behaviors,
            percentile_low=5.0,
            percentile_high=95.0,
            n_centroids=50,
        )
        print(f"[Init] Built {n_centroids} centroids", flush=True)

        # Add top programs to archive
        n_accepted = 0
        for prog in top_programs:
            child = Program(code=prog["code"], metadata={
                "execution_time": prog["execution_time"],
                "primary_score": prog["score"],
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

    async def run_generation():
        nonlocal generation, best_score, best_program, best_makespan, total_cost

        print(f"[Evolution] Starting evolution loop...", flush=True)
        loop = asyncio.get_event_loop()

        # Meta-advice tracking
        current_meta_advice = ""
        previous_meta_advice = ""
        period_metrics = {
            'total': 0,
            'timeouts': 0,
            'errors': 0,
            'rejections': 0,
            'acceptances': 0,
            'new_bests': 0,
            'error_messages': set(),
        }

        while total_cost < 5.0:
            generation += 1
            batch_start = time.time()

            # Every 10 generations, generate new meta-advice from accumulated metrics
            if generation > 1 and (generation - 1) % 10 == 0:
                # Get top 3 solutions from archive
                top_solutions = []
                try:
                    elites = list(pool._elites.values())
                    elites.sort(key=lambda e: e.result.primary_score, reverse=True)
                    for elite in elites[:3]:
                        score = elite.result.primary_score
                        code = elite.program.code
                        top_solutions.append((score, code))
                except Exception:
                    pass  # If we can't get elites, continue without them

                print(f"[Gen {generation}] Generating meta-advice with heavy model...", flush=True)
                current_meta_advice, advice_cost = await generate_meta_advice(
                    metrics=period_metrics,
                    previous_advice=previous_meta_advice,
                    best_score=best_score,
                    top_solutions=top_solutions,
                    model=HEAVY_MODEL,
                )
                total_cost += advice_cost
                previous_meta_advice = current_meta_advice
                # Reset metrics for next period
                period_metrics = {
                    'total': 0,
                    'timeouts': 0,
                    'errors': 0,
                    'rejections': 0,
                    'acceptances': 0,
                    'new_bests': 0,
                    'error_messages': set(),
                }
                print(f"[Gen {generation}] Meta-advice updated (cost: ${advice_cost:.4f})", flush=True)
                print(f"[Meta-Advice]\n{current_meta_advice}\n", flush=True)

            print(f"[Gen {generation}] Starting generation...", flush=True)

            # Determine which samplers to use this generation
            # All samplers run every generation; heavy models use HEAVY_MODEL
            # Format: (display_name, real_sampler_name, model)
            active_samplers = []
            for name in sampler_names:
                sampler = pool.get_sampler(name)
                if sampler.model_type == "heavy":
                    active_samplers.append((name, name, HEAVY_MODEL))
                elif name == "ucb":
                    # Run 3 parallel UCB samplers
                    for ucb_idx in range(3):
                        active_samplers.append((f"ucb_{ucb_idx}", "ucb", random.choice(LIGHT_MODELS)))
                else:
                    active_samplers.append((name, name, random.choice(LIGHT_MODELS)))

            # Build prompts for each active sampler
            prompts = []
            sampler_data = []
            for display_name, real_sampler, model in active_samplers:
                sample = pool.sample(real_sampler, n_parents=1 + n_inspirations)  # 1 parent + inspirations
                parents = [sample.parent] + sample.inspirations
                source_cell = sample.metadata.get("source_cell", 0)

                builder = PromptBuilder()
                builder.add_section("Problem", PROBLEM_DESCRIPTION, priority=10)
                builder.add_section("Signature", f"```python\n{FUNCTION_SIGNATURE}\n```", priority=20)
                builder.add_parents([ProgramWithScore(p, None) for p in parents], priority=30)
                builder.set_output_mode(OutputMode.FULL)  # Use FULL mode for better extraction

                # Add meta-advice at the end (low priority = appears last)
                if current_meta_advice:
                    builder.add_section("Meta-Advice", current_meta_advice, priority=100)

                prompts.append(builder.build())
                sampler_data.append({
                    "sampler": display_name,
                    "real_sampler": real_sampler,
                    "model": model,
                    "parents": parents,
                    "parent_code": parents[0].code,
                    "source_cell": source_cell,
                })

            # Generate for all samplers in parallel
            n_samplers = len(active_samplers)
            print(f"[Gen {generation}] LLM calls for {n_samplers} samplers...", flush=True)
            llm_tasks = [
                generate_for_island(i, prompts[i], sampler_data[i]["model"], 0.8)
                for i in range(n_samplers)
            ]
            results = await asyncio.gather(*llm_tasks)
            print(f"[Gen {generation}] All LLM calls complete", flush=True)

            # Extract code from LLM responses
            candidates = []
            for res in results:
                if "error" in res:
                    idx = res["island"]
                    print(f"[Gen {generation}] {sampler_data[idx]['sampler']} ERROR: {res['error'][:50]}", flush=True)
                    continue

                total_cost += res["cost"]
                idx = res["island"]
                tokens = res.get("tokens", 0)
                sname = sampler_data[idx]["sampler"]
                model = sampler_data[idx]["model"]

                new_code, validation_error = extract_and_validate_code(res["content"])

                if not new_code:
                    print(f"[Gen {generation}] {sname} VALIDATION FAIL ({model.split('/')[-1]}): {validation_error}", flush=True)
                    continue

                candidates.append({
                    "sampler": sname,
                    "real_sampler": sampler_data[idx]["real_sampler"],
                    "code": new_code,
                    "tokens": tokens,
                    "generation": generation,
                    "source_cell": sampler_data[idx]["source_cell"],
                    "model": model,
                })

            # Evaluate candidates with semaphore to ensure timeout starts when task runs
            EVAL_TIMEOUT = 300  # 5 minute timeout for txn scheduling
            print(f"[Gen {generation}] Evaluating {len(candidates)} candidates...", flush=True)
            eval_semaphore = asyncio.Semaphore(8)  # Match executor workers

            async def eval_with_timeout(idx, code):
                async with eval_semaphore:  # Timeout starts when semaphore acquired
                    start = time.time()
                    try:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(executor, evaluate_code_in_process, code),
                            timeout=EVAL_TIMEOUT
                        )
                        elapsed = time.time() - start
                        print(f"  [Eval] Candidate {idx} done in {elapsed:.1f}s", flush=True)
                        return idx, result
                    except asyncio.TimeoutError:
                        print(f"  [Eval] Candidate {idx} TIMEOUT after {EVAL_TIMEOUT}s", flush=True)
                        return idx, {"error": f"Timeout after {EVAL_TIMEOUT}s"}
                    except Exception as e:
                        print(f"  [Eval] Candidate {idx} ERROR: {e}", flush=True)
                        return idx, {"error": str(e)}

            eval_tasks = [eval_with_timeout(i, c["code"]) for i, c in enumerate(candidates)]
            indexed_results = await asyncio.gather(*eval_tasks)
            eval_results = [r for _, r in sorted(indexed_results, key=lambda x: x[0])]
            print(f"[Gen {generation}] All evaluations complete", flush=True)

            batch_time = time.time() - batch_start

            # Process results
            for cand, output in zip(candidates, eval_results):
                display_name = cand["sampler"]
                real_sampler = cand["real_sampler"]
                tokens = cand["tokens"]
                score = score_fn(output)
                exec_time = output.get("wall_time", 60.0)

                child = Program(code=cand["code"], metadata={
                    "execution_time": exec_time,
                    "primary_score": score,
                })
                eval_result = EvaluationResult(
                    program_id=child.id,
                    scores={'score': score},
                    is_valid="error" not in output,
                    error=output.get("error"),
                )

                # Track metrics for meta-advice
                period_metrics['total'] += 1

                if eval_result.is_valid:
                    accepted = pool.add(child, eval_result)
                    pool.update_sampler(real_sampler, cand["source_cell"], success=accepted, reward=score)
                    makespan = output.get("makespan", BASELINE)
                    wall_time = exec_time * 1000  # ms
                    status = "accepted" if accepted else "rejected"

                    if accepted:
                        period_metrics['acceptances'] += 1
                    else:
                        period_metrics['rejections'] += 1

                    if score > best_score:
                        best_score, best_program = score, child
                        best_makespan = makespan
                        status = "NEW BEST ★"
                        period_metrics['new_bests'] += 1

                    print(f"[Gen {generation}] {display_name:20s} {status:10s} | mkspan: {makespan:5.0f} | score: {score:5.1f} | time: {wall_time:6.0f}ms | best: {best_score:5.1f} | {tokens}tok | ${total_cost:.3f}", flush=True)
                else:
                    pool.update_sampler(real_sampler, cand["source_cell"], success=False)
                    err = eval_result.error[:30] if eval_result.error else "unknown"

                    # Track timeout vs other errors
                    if eval_result.error and "timeout" in eval_result.error.lower():
                        period_metrics['timeouts'] += 1
                    else:
                        period_metrics['errors'] += 1
                        # Add truncated error message to set (avoid duplicates)
                        if eval_result.error:
                            err_msg = eval_result.error[:100].strip()
                            period_metrics['error_messages'].add(err_msg)

                    print(f"[Gen {generation}] {display_name:20s} INVALID    | {err}...", flush=True)

            pool.on_generation_complete()
            print(f"    [Batch {generation} done in {batch_time:.1f}s | {n_samplers} samplers, archive: {pool.size()}]", flush=True)

            if generation % 25 == 0:
                print(f"{'─'*70}\n[MILESTONE] Best makespan: {best_makespan:.0f} | Score: {best_score:.1f}\n{'─'*70}", flush=True)

            # Save state after each generation
            save_state(
                generation=generation,
                pool=pool,
                best_score=best_score,
                best_program=best_program,
                total_cost=total_cost,
                extra_info={
                    "phase": "evolution",
                    "batch_time_s": batch_time,
                    "n_samplers": n_samplers,
                },
            )

        executor.shutdown(wait=False)

    async def main():
        await initialize_archive()
        await run_generation()

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
