#!/usr/bin/env python3
"""
AlphaEvolve on Transaction Scheduling Problem.
"""

import asyncio
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Add algoforge to path
ALGOFORGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ALGOFORGE_ROOT))

# Add txn_scheduling resources to path
TXN_RESOURCES = ALGOFORGE_ROOT.parent / "ADRS-Leaderboard" / "problems" / "txn_scheduling" / "resources"
sys.path.insert(0, str(TXN_RESOURCES))

import algoforge as af
from algoforge.core import Program, EvaluationResult
from algoforge.pool import CVTMAPElitesPool
from algoforge.budget import BudgetManager, BudgetExhausted
from algoforge.llm import LLMClient, LLMConfig, PromptBuilder, ProgramWithScore, OutputMode
from algoforge.behavior import BehaviorExtractor
from algoforge.behavior.extractor import FeatureVector
from algoforge.behavior.features import compute_code_length, compute_ast_depth, compute_math_operators
from algoforge.methods.alphaevolve import extract_code
import re
import numpy as np

def validate_python_syntax(code: str) -> tuple[bool, str]:
    """Check if code is valid Python syntax. Returns (is_valid, error_message)."""
    try:
        compile(code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"

def extract_replace_block(diff_response: str) -> str | None:
    """Extract just the REPLACE portion from a malformed diff response."""
    # Try to find content after ======= and before >>>>>>> REPLACE
    pattern = r'=======\s*(.*?)\s*>>>>>>> REPLACE'
    match = re.search(pattern, diff_response, re.DOTALL)
    if match:
        replace_content = match.group(1).strip()
        # Remove prompt artifacts like "# replacement code"
        lines = replace_content.split('\n')
        cleaned_lines = [l for l in lines if not l.strip().startswith('# replacement code')
                         and not l.strip().startswith('# exact code')]
        return '\n'.join(cleaned_lines).strip()
    return None

def apply_diff_debug(original: str, diff_response: str) -> tuple[str | None, str]:
    """Apply SEARCH/REPLACE diff blocks to original code. Returns (result, debug_info)."""
    result = original
    debug_lines = []

    # Primary pattern: standard SEARCH/REPLACE format
    pattern = r'<<<<<<< SEARCH\s*(.*?)\s*=======\s*(.*?)\s*>>>>>>> REPLACE'
    matches = re.findall(pattern, diff_response, re.DOTALL)

    if not matches:
        debug_lines.append(f"No proper SEARCH/REPLACE blocks found")

        # Check if it's a malformed diff (has ======= and >>>>>>> REPLACE but no <<<<<<< SEARCH)
        if '=======' in diff_response and '>>>>>>> REPLACE' in diff_response:
            debug_lines.append(f"Detected malformed diff (missing <<<<<<< SEARCH), trying to extract REPLACE block")
            replace_code = extract_replace_block(diff_response)
            if replace_code:
                is_valid, err = validate_python_syntax(replace_code)
                if is_valid:
                    debug_lines.append(f"Extracted REPLACE block ({len(replace_code)} chars), syntax valid")
                    return replace_code, "\n".join(debug_lines)
                else:
                    debug_lines.append(f"Extracted REPLACE block has syntax error: {err}")

        # Fallback to extract_code
        debug_lines.append(f"Falling back to extract_code")
        extracted = extract_code(diff_response)
        if extracted:
            # Validate syntax before accepting
            is_valid, err = validate_python_syntax(extracted)
            if is_valid:
                debug_lines.append(f"extract_code succeeded, got {len(extracted)} chars, syntax valid")
                return extracted, "\n".join(debug_lines)
            else:
                debug_lines.append(f"extract_code got {len(extracted)} chars but syntax error: {err}")
                # Check if the extracted code contains diff markers (common failure mode)
                if '=======' in extracted or '>>>>>>> REPLACE' in extracted:
                    debug_lines.append(f"Extracted code contains diff markers - this is the bug!")
                return None, "\n".join(debug_lines)
        else:
            debug_lines.append(f"extract_code also failed")
        return None, "\n".join(debug_lines)

    debug_lines.append(f"Found {len(matches)} SEARCH/REPLACE blocks")

    for i, (search, replace) in enumerate(matches):
        search_stripped = search.strip()
        replace_stripped = replace.strip()

        debug_lines.append(f"\n--- Block {i+1} ---")
        debug_lines.append(f"SEARCH (raw, {len(search)} chars): {repr(search[:100])}...")
        debug_lines.append(f"SEARCH (stripped, {len(search_stripped)} chars): {repr(search_stripped[:100])}...")
        debug_lines.append(f"REPLACE (stripped, {len(replace_stripped)} chars): {repr(replace_stripped[:100])}...")

        # Check if raw search exists
        if search in result:
            debug_lines.append(f"✓ Raw search found in result")
        else:
            debug_lines.append(f"✗ Raw search NOT found in result")

        # Check if stripped search exists
        if search_stripped in result:
            debug_lines.append(f"✓ Stripped search found in result")
            result = result.replace(search_stripped, replace_stripped, 1)
            debug_lines.append(f"Applied replacement")
        else:
            debug_lines.append(f"✗ Stripped search NOT found in result")
            debug_lines.append(f"First 200 chars of result: {repr(result[:200])}")

            # Fallback: If SEARCH doesn't match but we have a valid REPLACE,
            # just use the REPLACE as full code (if it's a complete program)
            if 'def get_random_costs' in replace_stripped:
                is_valid, err = validate_python_syntax(replace_stripped)
                if is_valid:
                    debug_lines.append(f"SEARCH mismatch but REPLACE is valid complete code, using it directly")
                    return replace_stripped, "\n".join(debug_lines)
                else:
                    debug_lines.append(f"REPLACE block has syntax error: {err}")

            return None, "\n".join(debug_lines)

    # Validate final result
    is_valid, err = validate_python_syntax(result)
    if not is_valid:
        debug_lines.append(f"Final result has syntax error: {err}")
        return None, "\n".join(debug_lines)

    return result, "\n".join(debug_lines)

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3
from prompts import PROBLEM_DESCRIPTION, FUNCTION_SIGNATURE, SEED_PROGRAM, SEED_INSPIRATIONS
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
    "openrouter/openai/gpt-5.1-codex-max": {
        "max_tokens": 32768,
        "max_input_tokens": 200000,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.00000125,  # $1.25/1M input
        "output_cost_per_token": 0.00001,    # $10/1M output
        "litellm_provider": "openrouter",
    },
    "openrouter/openai/gpt-5.1-codex-mini": {
        "max_tokens": 32768,
        "max_input_tokens": 200000,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.00000025,  # $0.25/1M input
        "output_cost_per_token": 0.000002,   # $2/1M output
        "litellm_provider": "openrouter",
    },
    "openrouter/google/gemini-2.5-flash": {
        "max_tokens": 65536,
        "max_input_tokens": 1048576,
        "max_output_tokens": 65536,
        "input_cost_per_token": 0.0000003,    # $0.30/1M input
        "output_cost_per_token": 0.0000025,   # $2.50/1M output
        "litellm_provider": "openrouter",
    },
    "openrouter/google/gemini-2.5-flash-lite": {
        "max_tokens": 32768,
        "max_input_tokens": 1048576,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0000001,    # $0.10/1M input
        "output_cost_per_token": 0.0000004,   # $0.40/1M output
        "litellm_provider": "openrouter",
    },
    "openrouter/openai/gpt-5.2": {
        "max_tokens": 65536,
        "max_input_tokens": 400000,
        "max_output_tokens": 65536,
        "input_cost_per_token": 0.00000175,  # $1.75/1M input
        "output_cost_per_token": 0.000014,   # $14/1M output
        "litellm_provider": "openrouter",
    },
    "openrouter/moonshotai/kimi-k2-0905": {
        "max_tokens": 32768,
        "max_input_tokens": 262144,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.00000039,  # $0.39/1M input
        "output_cost_per_token": 0.0000019,  # $1.90/1M output
        "litellm_provider": "openrouter",
    },
    "openrouter/x-ai/grok-4.1-fast": {
        "max_tokens": 32768,
        "max_input_tokens": 2000000,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0000002,   # $0.20/1M input
        "output_cost_per_token": 0.0000005,  # $0.50/1M output
        "litellm_provider": "openrouter",
    },
    "openrouter/x-ai/grok-code-fast-1": {
        "max_tokens": 32768,
        "max_input_tokens": 256000,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.0000002,   # $0.20/1M input
        "output_cost_per_token": 0.0000015,  # $1.50/1M output
        "litellm_provider": "openrouter",
    },
    "openrouter/z-ai/glm-4.6": {
        "max_tokens": 32768,
        "max_input_tokens": 204800,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.00000039,  # $0.39/1M input
        "output_cost_per_token": 0.0000019,  # $1.90/1M output
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

    from txn_simulator import Workload
    from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

    workloads, baseline, _ = _init_workloads()

    namespace = {
        'Workload': Workload, 'WORKLOAD_1': WORKLOAD_1,
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

        # Validate schedules
        for i, sched in enumerate(schedules):
            if set(sched) != set(range(workloads[i].num_txns)):
                return {"error": f"Invalid schedule for workload {i}"}

        return {
            "makespan": makespan,
            "algo_time": algo_time,
            "wall_time": wall_time,
        }
    except Exception as e:
        return {"error": str(e)}


# Initialize in main process
WORKLOADS, BASELINE, EFFECTIVE_OPTIMAL = None, None, None


def get_code_structure_features(code: str) -> dict[str, float]:
    """Compute code structure features using algoforge primitives."""
    import ast
    try:
        prog = Program(code=code, metadata={})
        tree = ast.parse(code)
        return {
            'code_length': compute_code_length(prog, tree),
            'ast_depth': compute_ast_depth(prog, tree),
            'math_operators': compute_math_operators(prog, tree),
        }
    except SyntaxError:
        return {'code_length': 0.0, 'ast_depth': 0.0, 'math_operators': 0.0}


class WorkloadPerfBehaviorExtractor:
    """Extracts walltime + code structure as behavior dimensions."""

    def __init__(self):
        # 4 behavior dimensions: walltime + 3 code structure
        self.features = ['walltime', 'code_length', 'ast_depth', 'math_operators']

    def extract(self, program: Program) -> FeatureVector:
        """Extract 4-dimensional behavior: walltime + code structure."""
        import ast

        # Walltime from metadata (normalized: faster = higher score)
        # Normalize walltime: 60s -> 0, 0s -> 1
        raw_walltime = program.metadata.get('walltime', 60.0)
        normalized_walltime = max(0, 1 - raw_walltime / 60.0)
        values = {
            'walltime': normalized_walltime,
        }

        # Code structure features using algoforge primitives
        try:
            tree = ast.parse(program.code)
            values['code_length'] = compute_code_length(program, tree)
            values['ast_depth'] = compute_ast_depth(program, tree)
            values['math_operators'] = compute_math_operators(program, tree)
        except SyntaxError:
            values['code_length'] = 0.0
            values['ast_depth'] = 0.0
            values['math_operators'] = 0.0

        return FeatureVector(values)


def compute_score(makespan: float) -> float:
    """Compute 0-100 score using leaderboard formula."""
    if makespan >= BASELINE:
        return 0.0
    if makespan <= EFFECTIVE_OPTIMAL:
        return 100.0
    return ((BASELINE - makespan) / (BASELINE - EFFECTIVE_OPTIMAL)) * 100


def score_fn(output: dict, inp: dict, exec_time: float) -> float:
    """Score using leaderboard formula (0-100, higher = better)."""
    if output is None or "error" in output:
        return 0.0
    makespan = output.get("makespan", BASELINE)
    return compute_score(makespan)


# Provider routing for specific models
PROVIDER_ROUTING = {}


async def generate_for_island(island_idx: int, prompt: str, model: str, temperature: float) -> dict:
    """Generate code for one island asynchronously."""
    start = time.time()
    extra = PROVIDER_ROUTING.get(model, {})
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=30000,  # Increased for longer code generation
            timeout=180,  # 3 minute timeout for longer responses
            extra_body=extra if extra else None,
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

    # Model configuration: light vs heavy
    LIGHT_MODELS = [
        'openrouter/google/gemini-2.5-flash-lite',
    ]
    HEAVY_MODEL = 'openrouter/google/gemini-2.5-flash'

    budget = BudgetManager(max_llm_cost=5.0)
    n_workers = 4
    n_inspirations = 1  # Number of inspiration programs to use alongside parent

    extractor = WorkloadPerfBehaviorExtractor()

    # Single archive with deferred centroid initialization
    pool = CVTMAPElitesPool(
        behavior_extractor=extractor,
        n_centroids=50,  # Will be set from data
        subscore_keys=["walltime"],
        defer_centroids=True,  # Build centroids from initial LLM generations
    )

    # Evaluate only the main seed program
    print(f"Evaluating seed program...")

    seed_output = evaluate_code_in_process(SEED_PROGRAM)
    seed_score = score_fn(seed_output, {}, 0)
    best_makespan = seed_output.get("makespan", BASELINE)
    print(f"  Seed: makespan={best_makespan:.0f}, score={seed_score:.1f}")

    seed_code_features = get_code_structure_features(SEED_PROGRAM)
    seed_program = Program(code=SEED_PROGRAM, metadata={
        "walltime": seed_output.get("wall_time", 60.0),
        **seed_code_features,
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

    # Create process pool once (expensive to create)
    executor = ProcessPoolExecutor(max_workers=n_workers)

    async def initialize_archive():
        """Generate 60 solutions, build centroids from behaviors, then fill archive."""
        nonlocal total_cost, best_score, best_program, best_makespan

        n_init = 50  # Match n_centroids for initial population
        print(f"\n[Init] Generating {n_init} initial solutions with light models...")
        loop = asyncio.get_event_loop()

        # Use seed program plus inspirations
        all_seed_codes = [SEED_PROGRAM] + SEED_INSPIRATIONS[:n_inspirations]
        all_seed_programs = [Program(code=c, metadata={}) for c in all_seed_codes]

        # Build prompts sampling from different seeds
        prompts = []
        init_parent_codes = []  # Track which seed code was used for diff
        for i in range(n_init):
            parent_prog = all_seed_programs[i % len(all_seed_programs)]
            builder = PromptBuilder()
            builder.add_section("Problem", PROBLEM_DESCRIPTION, priority=10)
            builder.add_section("Signature", f"```python\n{FUNCTION_SIGNATURE}\n```", priority=20)
            builder.add_parents([ProgramWithScore(parent_prog, None)], priority=30)
            builder.set_output_mode(OutputMode.DIFF)
            prompts.append(builder.build())
            init_parent_codes.append(parent_prog.code)

        # Make parallel LLM calls (alternate between light models)
        llm_tasks = [
            generate_for_island(i, prompts[i], LIGHT_MODELS[i % len(LIGHT_MODELS)], 0.8)
            for i in range(n_init)
        ]
        results = await asyncio.gather(*llm_tasks)
        print(f"[Init] All {n_init} LLM calls complete")

        # Extract code from responses
        candidates = []
        for res in results:
            if "error" in res:
                print(f"[Init] Candidate {res['island']} ERROR: {res['error'][:50]}")
                continue

            total_cost += res["cost"]
            idx = res["island"]

            has_diff = "<<<<<<< SEARCH" in res["content"]
            new_code, _ = apply_diff_debug(init_parent_codes[idx], res["content"])
            tokens = res.get("tokens", 0)

            if not new_code:
                print(f"[Init] Candidate {idx} EXTRACT FAIL | {tokens} tok")
                continue

            candidates.append({"idx": idx, "code": new_code, "tokens": tokens})

        print(f"[Init] Evaluating {len(candidates)} candidates...")

        # Evaluate all in parallel
        async def eval_candidate(idx, code):
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor, evaluate_code_in_process, code),
                    timeout=300
                )
                return idx, result
            except asyncio.TimeoutError:
                return idx, {"error": "Timeout"}
            except Exception as e:
                return idx, {"error": str(e)}

        eval_tasks = [eval_candidate(c["idx"], c["code"]) for c in candidates]
        eval_results = await asyncio.gather(*eval_tasks)
        eval_map = {idx: res for idx, res in eval_results}

        # Collect valid programs and their behavior vectors (6 dimensions)
        valid_programs = []
        behavior_vectors = []
        n_errors = 0
        for cand in candidates:
            output = eval_map.get(cand["idx"], {"error": "missing"})
            if "error" in output:
                n_errors += 1
                if n_errors <= 5:  # Show first 5 errors
                    print(f"[Init] Candidate {cand['idx']} EVAL FAIL: {output['error'][:80]}")
                continue
            if True:  # was: if "error" not in output
                score = score_fn(output, {}, 0)
                walltime = output.get("wall_time", 60.0)
                normalized_walltime = max(0, 1 - walltime / 60.0)
                code_feats = get_code_structure_features(cand["code"])
                valid_programs.append({
                    "code": cand["code"],
                    "score": score,
                    "output": output,
                    "walltime": walltime,
                    **code_feats,
                })
                behavior_vectors.append(np.array([
                    normalized_walltime,
                    code_feats['code_length'],
                    code_feats['ast_depth'],
                    code_feats['math_operators'],
                ]))

        # Add seed's behavior vector (4 dimensions)
        seed_walltime = seed_program.metadata.get("walltime", 60.0)
        seed_normalized_walltime = max(0, 1 - seed_walltime / 60.0)
        seed_behavior = np.array([
            seed_normalized_walltime,
            seed_program.metadata.get("code_length", 0),
            seed_program.metadata.get("ast_depth", 0),
            seed_program.metadata.get("math_operators", 0),
        ])
        behavior_vectors.append(seed_behavior)

        # Build centroids from behavior vectors using k-means (ignoring 5th and 95th percentile outliers)
        n_centroids = pool.set_centroids_from_data(
            behavior_vectors,
            percentile_low=5.0,
            percentile_high=95.0,
            n_centroids=50,  # k-means will spread 50 centroids across the behavior space
        )
        print(f"[Init] Built {n_centroids} centroids from {len(behavior_vectors)} behavior vectors")

        # Now add seed program to archive
        seed_result = EvaluationResult(
            program_id=seed_program.id,
            scores={'score': best_score},
            is_valid=True,
        )
        pool.add(seed_program, seed_result)

        # Add all valid programs to archive
        n_accepted = 1  # seed already added
        for prog in valid_programs:
            child = Program(code=prog["code"], metadata={
                "walltime": prog["walltime"],
                "code_length": prog["code_length"],
                "ast_depth": prog["ast_depth"],
                "math_operators": prog["math_operators"],
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

        print(f"[Init] Done: {n_accepted} accepted, archive size: {pool.size()}, best: {best_score:.1f}, cost: ${total_cost:.3f}\n")

    async def run_generation():
        nonlocal generation, best_score, best_program, best_makespan, total_cost

        loop = asyncio.get_event_loop()
        timeout_feedback = None

        while total_cost < 5.0:
            generation += 1
            batch_start = time.time()

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
                builder.set_output_mode(OutputMode.DIFF)

                if timeout_feedback:
                    builder.add_section("Critical Feedback", timeout_feedback, priority=5)

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
            print(f"[Gen {generation}] LLM calls for {n_samplers} samplers...")
            llm_tasks = [
                generate_for_island(i, prompts[i], sampler_data[i]["model"], 0.8)
                for i in range(n_samplers)
            ]
            results = await asyncio.gather(*llm_tasks)
            print(f"[Gen {generation}] All LLM calls complete")

            # Extract code from LLM responses
            candidates = []
            for res in results:
                if "error" in res:
                    idx = res["island"]
                    print(f"[Gen {generation}] {sampler_data[idx]['sampler']} ERROR: {res['error'][:50]}")
                    continue

                total_cost += res["cost"]
                idx = res["island"]
                parent_code = sampler_data[idx]["parent_code"]

                has_diff = "<<<<<<< SEARCH" in res["content"]
                new_code, diff_debug = apply_diff_debug(parent_code, res["content"])
                tokens = res.get("tokens", 0)
                mode = "DIFF" if has_diff else "FULL"

                if not new_code:
                    sname = sampler_data[idx]["sampler"]
                    print(f"[Gen {generation}] {sname} EXTRACT FAIL | {mode} | {tokens} tok")
                    print(f"{'─'*60}")
                    print(f"[DEBUG] Diff analysis:\n{diff_debug}")
                    print(f"{'─'*60}")
                    print(f"[DEBUG] Parent code (first 500 chars):\n{parent_code[:500]}")
                    print(f"{'─'*60}")
                    print(f"[DEBUG] LLM Response:\n{res['content']}")
                    print(f"{'─'*60}")
                    continue

                candidates.append({
                    "sampler": sampler_data[idx]["sampler"],
                    "real_sampler": sampler_data[idx]["real_sampler"],
                    "code": new_code,
                    "tokens": tokens,
                    "mode": mode,
                    "generation": generation,
                    "source_cell": sampler_data[idx]["source_cell"],
                })

            # Evaluate candidates
            EVAL_TIMEOUT = 300
            print(f"[Gen {generation}] Evaluating {len(candidates)} candidates...")

            async def eval_with_timeout(idx, code):
                start = time.time()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(executor, evaluate_code_in_process, code),
                        timeout=EVAL_TIMEOUT
                    )
                    elapsed = time.time() - start
                    print(f"  [Eval] Candidate {idx} done in {elapsed:.1f}s")
                    return idx, result
                except asyncio.TimeoutError:
                    print(f"  [Eval] Candidate {idx} TIMEOUT after {EVAL_TIMEOUT}s")
                    return idx, {"error": f"Timeout after {EVAL_TIMEOUT}s"}
                except Exception as e:
                    print(f"  [Eval] Candidate {idx} ERROR: {e}")
                    return idx, {"error": str(e)}

            eval_tasks = [eval_with_timeout(i, c["code"]) for i, c in enumerate(candidates)]
            indexed_results = await asyncio.gather(*eval_tasks)
            eval_results = [r for _, r in sorted(indexed_results, key=lambda x: x[0])]
            print(f"[Gen {generation}] All evaluations complete")

            batch_time = time.time() - batch_start

            # Process results
            for cand, output in zip(candidates, eval_results):
                display_name = cand["sampler"]
                real_sampler = cand["real_sampler"]
                tokens = cand["tokens"]
                mode = cand["mode"]
                score = score_fn(output, {}, 0)

                code_feats = get_code_structure_features(cand["code"])
                child_metadata = {
                    "walltime": output.get("wall_time", 60.0),
                    **code_feats,
                }
                child = Program(code=cand["code"], metadata=child_metadata)
                eval_result = EvaluationResult(
                    program_id=child.id,
                    scores={'score': score},
                    is_valid="error" not in output,
                    error=output.get("error"),
                )

                if eval_result.is_valid:
                    accepted = pool.add(child, eval_result)
                    # Update sampler statistics (use real_sampler for stats)
                    pool.update_sampler(real_sampler, cand["source_cell"], success=accepted, reward=score)
                    makespan = output.get("makespan", BASELINE)
                    status = "accepted" if accepted else "rejected"
                    if score > best_score:
                        best_score, best_program = score, child
                        best_makespan = makespan
                        status = "NEW BEST ★"
                    print(f"[Gen {generation}] {display_name:20s} {status:10s} | mkspan: {makespan:5.0f} | score: {score:5.1f} | best: {best_score:5.1f} | {mode} {tokens}tok | ${total_cost:.3f}")
                else:
                    pool.update_sampler(real_sampler, cand["source_cell"], success=False)
                    err = eval_result.error[:30] if eval_result.error else "unknown"
                    print(f"[Gen {generation}] {display_name:20s} INVALID    | {err}...")
                    if "syntax" in (eval_result.error or "").lower():
                        print(f"{'─'*60}")
                        print(f"[DEBUG] Code with syntax error (mode: {mode}):\n{cand['code']}")
                        print(f"{'─'*60}")

            pool.on_generation_complete()
            print(f"    [Batch {generation} done in {batch_time:.1f}s | {n_samplers} samplers, archive: {pool.size()}]")

            # Check for timeout feedback
            n_timeouts = sum(1 for r in eval_results if "error" in r and "Timeout" in r.get("error", ""))
            if candidates and n_timeouts == len(candidates):
                timeout_feedback = (
                    "Note: Previous candidates exceeded the 300s time limit. "
                    "Consider simpler approaches with lower computational complexity."
                )
                print(f"    [FEEDBACK] All {n_timeouts} candidates timed out")
            else:
                timeout_feedback = None

            if generation % 25 == 0:
                print(f"{'─'*70}\n[MILESTONE] Best makespan: {best_makespan:.0f} | Score: {best_score:.1f}\n{'─'*70}")

        executor.shutdown(wait=False)

    async def main():
        await initialize_archive()
        await run_generation()

    asyncio.run(main())

    print(f"\n{'='*70}")
    print(f"Complete | Generations: {generation}")
    print(f"Best makespan: {best_makespan:.0f} | Score: {best_score:.1f}/100")
    print(f"{'='*70}\n")

    out = Path(__file__).parent / "best_solution.py"
    out.write_text(best_program.code)
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
