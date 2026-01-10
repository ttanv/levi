#!/usr/bin/env python3
"""
AlphaEvolve on EPLB (Expert Parallelism Load Balancer) Problem.
"""

import asyncio
import sys
import time
from pathlib import Path
from concurrent.futures import BrokenExecutor
import multiprocessing

# Add algoforge to path (must be before algoforge imports)
ALGOFORGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ALGOFORGE_ROOT))

from algoforge.utils import ResilientProcessPool

# Add eplb resources to path
EPLB_RESOURCES = ALGOFORGE_ROOT.parent / "ADRS-Leaderboard" / "problems" / "eplb"
sys.path.insert(0, str(EPLB_RESOURCES))

import algoforge as af
from algoforge.core import Program, EvaluationResult
from algoforge.pool import CVTMAPElitesPool
from algoforge.budget import BudgetManager, BudgetExhausted
from algoforge.llm import LLMClient, LLMConfig, PromptBuilder, ProgramWithScore, OutputMode
from algoforge.behavior import BehaviorExtractor
from algoforge.behavior.extractor import FeatureVector
from algoforge.utils import extract_code
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
    # Check for CUDA usage
    cuda_patterns = ['.cuda()', 'torch.cuda', '.to("cuda")', ".to('cuda')"]
    for pattern in cuda_patterns:
        if pattern in code:
            return False, f"CUDA usage detected: {pattern}"

    # Check for missing function definition
    if 'def rebalance_experts' not in code:
        return False, "Missing rebalance_experts function"

    # Check for import torch
    if 'import torch' not in code:
        return False, "Missing import torch"

    # Quick execution test with small tensors
    try:
        import torch
        namespace = {'torch': torch}
        exec(code, namespace)

        if 'rebalance_experts' not in namespace:
            return False, "rebalance_experts not defined after exec"

        # Test with small dummy input
        test_weight = torch.ones(2, 256)  # 2 layers, 256 experts
        phy2log, log2phy, logcnt = namespace['rebalance_experts'](
            test_weight, 288, 8, 4, 32
        )

        # Validate shapes
        if phy2log.shape != (2, 288):
            return False, f"phy2log shape {phy2log.shape} != (2, 288)"
        if logcnt.shape != (2, 256):
            return False, f"logcnt shape {logcnt.shape} != (2, 256)"
        if log2phy.shape[0] != 2 or log2phy.shape[1] != 256:
            return False, f"log2phy shape {log2phy.shape} invalid"

        # Validate logcnt sums to 288
        sums = logcnt.sum(dim=1)
        if not torch.all(sums == 288):
            return False, f"logcnt sums {sums.tolist()} != 288"

        # Validate phy2log values in range
        if phy2log.min() < 0 or phy2log.max() >= 256:
            return False, f"phy2log values out of range [0, 255]: min={phy2log.min()}, max={phy2log.max()}"

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
        if 'def rebalance_experts' in match:
            candidates.append(match.strip())

    # Try raw extraction if no code blocks
    if not candidates and 'def rebalance_experts' in response:
        lines = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if 'import torch' in line or 'def rebalance_experts' in line:
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

        # Quick validation with small tensors
        is_valid, validation_err = quick_validate_code(candidate)
        if is_valid:
            return candidate, ""
        else:
            last_error = validation_err

    return None, last_error


def extract_code(response: str) -> str | None:
    """Extract Python code from LLM response. Returns code or None."""
    code, _ = extract_and_validate_code(response)
    return code

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
    "openrouter/z-ai/glm-4.5-air": {
        "max_tokens": 32768,
        "max_input_tokens": 131072,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.000000104,  # $0.104/1M input
        "output_cost_per_token": 0.00000068,  # $0.68/1M output
        "litellm_provider": "openrouter",
    },
    "openrouter/openai/gpt-oss-120b": {
        "max_tokens": 131072,
        "max_input_tokens": 131072,
        "max_output_tokens": 131072,
        "input_cost_per_token": 0.000000039,  # $0.039/1M input
        "output_cost_per_token": 0.00000019,  # $0.19/1M output
        "litellm_provider": "openrouter",
    },
    "openrouter/deepseek/deepseek-v3.2": {
        "max_tokens": 163840,
        "max_input_tokens": 163840,
        "max_output_tokens": 163840,
        "input_cost_per_token": 0.00000026,  # $0.26/1M input
        "output_cost_per_token": 0.00000038,  # $0.38/1M output
        "litellm_provider": "openrouter",
    },
    "openrouter/qwen/qwen3-coder-30b-a3b-instruct": {
        "max_tokens": 32768,
        "max_input_tokens": 160000,
        "max_output_tokens": 32768,
        "input_cost_per_token": 0.00000007,  # $0.07/1M input
        "output_cost_per_token": 0.00000027,  # $0.27/1M output
        "litellm_provider": "openrouter",
    },
})

# EPLB Constants
NUM_REPLICAS = 288
NUM_GROUPS = 8
NUM_GPUS = 32
NUM_NODES = 4
REBALANCE_INTERVAL = 100

# Pre-load workloads (lazy init for multiprocessing)
_WORKLOADS = None


def _init_workloads():
    """Initialize workloads (called once per process)."""
    global _WORKLOADS
    if _WORKLOADS is None:
        import json
        import torch
        import os

        # Find workload file
        docker_path = "/datasets/eplb/expert-load.json"
        local_path = EPLB_RESOURCES.parent.parent / "datasets" / "eplb" / "expert-load.json"

        if os.path.exists(docker_path):
            workload_path = docker_path
        elif local_path.exists():
            workload_path = str(local_path)
        else:
            raise FileNotFoundError(f"Workload file not found. Tried: {docker_path}, {local_path}")

        with open(workload_path, "r") as f:
            data = json.load(f)

        total_len = len(data['load_history'])
        workloads = []
        for i in range(0, total_len, REBALANCE_INTERVAL):
            start = i
            end = min(start + REBALANCE_INTERVAL, total_len)
            load = torch.tensor([x['logical_expert_load'] for x in data['load_history'][start:end]]).sum(dim=0)
            workloads.append(load)

        _WORKLOADS = workloads
    return _WORKLOADS


def simulate_inference(log2phy, logcnt, workload):
    """Simulate MoE inference and return balancedness scores."""
    import torch

    num_layers, num_logical_experts = workload.shape

    # Initialize physical expert load accumulator
    num_physical_experts = NUM_REPLICAS
    total_physical_load = torch.zeros(num_layers, num_physical_experts, dtype=torch.float, device=workload.device)

    # For each logical expert, distribute load to its physical replicas
    for layer_id in range(num_layers):
        for logical_id in range(num_logical_experts):
            logical_load = workload[layer_id][logical_id].item()
            if logical_load <= 0:
                continue

            num_replicas = int(logcnt[layer_id][logical_id].item())
            if num_replicas <= 0:
                continue

            physical_ids = log2phy[layer_id][logical_id][:num_replicas]
            replica_load = logical_load / num_replicas
            total_physical_load[layer_id, physical_ids] += replica_load

    # Calculate balancedness
    total_load = total_physical_load.sum()
    if total_load == 0:
        return 0.0, 0.0

    # Compute expert load
    expert_layer_avg = total_physical_load.mean(dim=1).sum().item()
    expert_layer_max = total_physical_load.max(dim=1).values.sum().item()
    balancedness_expert = expert_layer_avg / expert_layer_max if expert_layer_max > 0 else 0.0

    # Compute GPU load
    gpu_load = total_physical_load.view(num_layers, NUM_GPUS, -1).sum(dim=2)

    layer_avg = gpu_load.mean(dim=1)
    layer_max = gpu_load.max(dim=1).values

    avg_load = layer_avg.sum().item()
    max_load = layer_max.sum().item()

    balancedness_gpu = avg_load / max_load if max_load > 0 else 0.0

    return balancedness_gpu, balancedness_expert


def evaluate_code_in_process(code: str) -> dict:
    """Execute EPLB code in a subprocess. Must be picklable."""
    import torch
    import time as time_module

    # Ensure paths are set up in worker process
    import sys
    from pathlib import Path
    algoforge_root = Path(__file__).resolve().parents[2]
    eplb_resources = algoforge_root.parent / "ADRS-Leaderboard" / "problems" / "eplb"
    if str(algoforge_root) not in sys.path:
        sys.path.insert(0, str(algoforge_root))
    if str(eplb_resources) not in sys.path:
        sys.path.insert(0, str(eplb_resources))

    workloads = _init_workloads()

    namespace = {
        'torch': torch,
        'time': time_module,
        '__builtins__': __builtins__,
    }

    try:
        exec(code, namespace)
        if 'rebalance_experts' not in namespace:
            return {"error": "rebalance_experts not defined"}

        balancedness_scores_gpu = []
        balancedness_scores_expert = []
        times_algorithm = []

        for i in range(len(workloads) - 1):
            start_time = time_module.perf_counter()
            _, log2phy, logcnt = namespace['rebalance_experts'](
                workloads[i],
                NUM_REPLICAS,
                NUM_GROUPS,
                NUM_NODES,
                NUM_GPUS,
            )
            end_time_algorithm = time_module.perf_counter()

            balancedness_gpu, balancedness_expert = simulate_inference(
                log2phy, logcnt, workloads[i + 1]
            )

            balancedness_scores_gpu.append(balancedness_gpu)
            balancedness_scores_expert.append(balancedness_expert)
            times_algorithm.append(end_time_algorithm - start_time)

        avg_balancedness_gpu = sum(balancedness_scores_gpu) / len(balancedness_scores_gpu)
        avg_balancedness_expert = sum(balancedness_scores_expert) / len(balancedness_scores_expert)
        avg_time_algorithm = sum(times_algorithm) / len(times_algorithm)

        # Scoring formula from evaluator
        balancedness_score = avg_balancedness_gpu * 90
        speed_raw = 0.002 / avg_time_algorithm if avg_time_algorithm > 0 else 2.0
        speed_capped = min(speed_raw, 2.0)
        speed_score = speed_capped * 5

        if avg_time_algorithm > 0.01:  # > 10ms
            slow_penalty = min(avg_time_algorithm * 20, 20)
        else:
            slow_penalty = 0

        score = balancedness_score + speed_score - slow_penalty

        return {
            "score": score,
            "balancedness_gpu": avg_balancedness_gpu,
            "balancedness_expert": avg_balancedness_expert,
            "avg_time": avg_time_algorithm,
            "balancedness_score": balancedness_score,
            "speed_score": speed_score,
            "slow_penalty": slow_penalty,
        }
    except Exception as e:
        return {"error": str(e)}


# Initialize in main process
WORKLOADS = None




def compute_score(output: dict) -> float:
    """Extract score from evaluation output."""
    if output is None or "error" in output:
        return 0.0
    return output.get("score", 0.0)


def score_fn(output: dict, inp: dict, exec_time: float) -> float:
    """Score using EPLB formula (higher = better)."""
    return compute_score(output)


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


# Provider routing for specific models
PROVIDER_ROUTING = {
    "openrouter/z-ai/glm-4.5-air": {
        "provider": {
            "order": ["Nebius Token Factory"],
            "allow_fallbacks": True,
        }
    },
}


async def generate_for_island(island_idx: int, prompt: str, model: str, temperature: float) -> dict:
    """Generate code for one island asynchronously."""
    start = time.time()
    extra = PROVIDER_ROUTING.get(model, {}).copy() if model in PROVIDER_ROUTING else {}

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
    global WORKLOADS
    WORKLOADS = _init_workloads()

    # Model configuration: multiple cheap light models for diversity
    LIGHT_MODELS = [
        'openrouter/qwen/qwen3-coder-30b-a3b-instruct',  # $0.07/$0.27 per M
        'openrouter/google/gemini-2.5-flash-lite',       # $0.10/$0.40 per M
        'openrouter/deepseek/deepseek-v3.2',             # $0.26/$0.38 per M
    ]
    HEAVY_MODEL = 'openrouter/google/gemini-2.5-flash'   # Better quality for exploration

    budget = BudgetManager(max_llm_cost=5.0)
    n_workers = 8
    n_inspirations = 1  # Number of inspiration programs to use alongside parent

    # Use behavior extractor with AST and score features
    extractor = BehaviorExtractor(
        ast_features=['loop_count', 'branch_count', 'function_count'],
        score_keys=['execution_time', 'score'],
    )

    # Single archive with deferred centroid initialization
    pool = CVTMAPElitesPool(
        behavior_extractor=extractor,
        n_centroids=50,  # Will be set from data
        subscore_keys=["execution_time"],
        defer_centroids=True,  # Build centroids from initial LLM generations
    )

    # Evaluate only the main seed program
    print(f"Evaluating seed program...")

    seed_output = evaluate_code_in_process(SEED_PROGRAM)
    seed_score = score_fn(seed_output, {}, 0)
    print(f"  Seed: score={seed_score:.1f}, balancedness={seed_output.get('balancedness_gpu', 0):.4f}, time={seed_output.get('avg_time', 0)*1000:.2f}ms")

    seed_program = Program(code=SEED_PROGRAM, metadata={
        "execution_time": seed_output.get("avg_time", 60.0),
        "primary_score": seed_score,
    })
    best_score = seed_score
    best_program = seed_program

    # Get sampler names for display
    sampler_names = pool.get_sampler_names()

    print(f"{'='*70}")
    print(f"AlphaEvolve - EPLB (Expert Parallelism Load Balancer)")
    print(f"{'='*70}")
    print(f"  Seed score:           {best_score:.1f}")
    print(f"  Budget:               $5.00")
    print(f"  Samplers:             {', '.join(sampler_names)} (UCB runs 3x)")
    print(f"  Light models:         {', '.join(LIGHT_MODELS)}")
    print(f"  Heavy model:          {HEAVY_MODEL} (every gen)")
    print(f"{'='*70}\n")

    generation = 0
    total_cost = 0.0

    # Create resilient process pool (auto-recovers if workers crash from OOM, segfault, etc.)
    executor = ResilientProcessPool(max_workers=n_workers, max_tasks_per_child=5)

    async def initialize_archive():
        """Generate 60 solutions, build centroids from behaviors, then fill archive."""
        nonlocal total_cost, best_score, best_program

        n_init = 100  # Larger initial population for more archive diversity
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
            builder.set_output_mode(OutputMode.FULL)
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
            tokens = res.get("tokens", 0)
            model = res.get("model", "unknown")

            new_code, validation_error = extract_and_validate_code(res["content"])

            if not new_code:
                n_extract_fail = getattr(initialize_archive, 'n_extract_fail', 0) + 1
                initialize_archive.n_extract_fail = n_extract_fail
                if n_extract_fail <= 5:  # Show first 5 validation errors
                    print(f"[Init] Candidate {idx} VALIDATION FAIL ({model}): {validation_error}", flush=True)
                continue

            candidates.append({"idx": idx, "code": new_code, "tokens": tokens, "model": model})

        print(f"[Init] Evaluating {len(candidates)} candidates...", flush=True)

        # Evaluate all in parallel with progress tracking
        eval_map = {}
        completed = 0

        async def eval_candidate(idx, code):
            nonlocal completed
            start = time.time()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor.executor, evaluate_code_in_process, code),
                    timeout=30
                )
                elapsed = time.time() - start
                completed += 1
                print(f"  [Init Eval] {completed}/{len(candidates)} Candidate {idx} done in {elapsed:.1f}s", flush=True)
                return idx, result
            except asyncio.TimeoutError:
                completed += 1
                print(f"  [Init Eval] {completed}/{len(candidates)} Candidate {idx} TIMEOUT", flush=True)
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

        # Collect valid programs and their behavior vectors (4 dimensions)
        valid_programs = []
        behavior_vectors = []
        n_errors = 0
        for cand in candidates:
            output = eval_map.get(cand["idx"], {"error": "missing"})
            if "error" in output:
                n_errors += 1
                if n_errors <= 5:  # Show first 5 errors
                    print(f"[Init] Candidate {cand['idx']} EVAL FAIL: {output['error'][:80]}", flush=True)
                continue
            if True:  # was: if "error" not in output
                score = score_fn(output, {}, 0)
                execution_time = output.get("avg_time", 60.0)
                valid_programs.append({
                    "code": cand["code"],
                    "score": score,
                    "output": output,
                    "execution_time": execution_time,
                })
                # Extract behavior vector using the extractor
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

        # Add seed's behavior vector using the extractor
        print(f"[Init] Extracting seed behavior...", flush=True)
        seed_behavior = extractor.extract(seed_program)
        behavior_vectors.append(np.array([seed_behavior[f] for f in extractor.features]))

        # Build centroids from behavior vectors using k-means (ignoring 5th and 95th percentile outliers)
        print(f"[Init] Building centroids from {len(behavior_vectors)} behavior vectors...", flush=True)
        n_centroids = pool.set_centroids_from_data(
            behavior_vectors,
            percentile_low=5.0,
            percentile_high=95.0,
            n_centroids=50,  # k-means will spread 50 centroids across the behavior space
        )
        print(f"[Init] Built {n_centroids} centroids", flush=True)

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
        print(f"[Init] State saved to {STATE_FILE}", flush=True)

    async def run_generation():
        nonlocal generation, best_score, best_program, total_cost

        print(f"[Evolution] Starting evolution loop...", flush=True)
        loop = asyncio.get_event_loop()
        timeout_feedback = None

        while total_cost < 5.0:
            generation += 1
            batch_start = time.time()
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
                builder.set_output_mode(OutputMode.FULL)

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

            # Evaluate candidates
            EVAL_TIMEOUT = 30
            print(f"[Gen {generation}] Evaluating {len(candidates)} candidates...")

            async def eval_with_timeout(idx, code):
                start = time.time()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(executor.executor, evaluate_code_in_process, code),
                        timeout=EVAL_TIMEOUT
                    )
                    elapsed = time.time() - start
                    print(f"  [Eval] Candidate {idx} done in {elapsed:.1f}s")
                    return idx, result
                except asyncio.TimeoutError:
                    print(f"  [Eval] Candidate {idx} TIMEOUT after {EVAL_TIMEOUT}s")
                    return idx, {"error": f"Timeout after {EVAL_TIMEOUT}s"}
                except BrokenExecutor as e:
                    # Pool crashed - recreate it and return error for this candidate
                    executor._recreate_executor()
                    print(f"  [Eval] Candidate {idx} POOL CRASHED (recreated)")
                    return idx, {"error": "Pool crashed, recreated"}
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
                score = score_fn(output, {}, 0)

                child_metadata = {
                    "execution_time": output.get("avg_time", 60.0),
                    "primary_score": score,
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
                    balancedness = output.get("balancedness_gpu", 0)
                    avg_time = output.get("avg_time", 0) * 1000  # ms
                    status = "accepted" if accepted else "rejected"
                    if score > best_score:
                        best_score, best_program = score, child
                        status = "NEW BEST ★"
                    print(f"[Gen {generation}] {display_name:20s} {status:10s} | score: {score:5.1f} | bal: {balancedness:.4f} | time: {avg_time:6.2f}ms | best: {best_score:5.1f} | {tokens}tok | ${total_cost:.3f}")
                else:
                    pool.update_sampler(real_sampler, cand["source_cell"], success=False)
                    err = eval_result.error[:30] if eval_result.error else "unknown"
                    print(f"[Gen {generation}] {display_name:20s} INVALID    | {err}...")
                    if "syntax" in (eval_result.error or "").lower():
                        print(f"{'─'*60}")
                        print(f"[DEBUG] Code with syntax error:\n{cand['code']}")
                        print(f"{'─'*60}")

            pool.on_generation_complete()
            print(f"    [Batch {generation} done in {batch_time:.1f}s | {n_samplers} samplers, archive: {pool.size()}]")

            # Check for timeout feedback
            n_timeouts = sum(1 for r in eval_results if "error" in r and "Timeout" in r.get("error", ""))
            if candidates and n_timeouts == len(candidates):
                timeout_feedback = (
                    "Note: Previous candidates exceeded the 30s time limit. "
                    "Consider simpler approaches with lower computational complexity."
                )
                print(f"    [FEEDBACK] All {n_timeouts} candidates timed out")
            else:
                timeout_feedback = None

            if generation % 25 == 0:
                print(f"{'─'*70}\n[MILESTONE] Best score: {best_score:.1f}\n{'─'*70}")

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
                    "n_timeouts": n_timeouts if candidates else 0,
                },
            )

        executor.shutdown(wait=False)

    async def main():
        await initialize_archive()
        await run_generation()

    asyncio.run(main())

    print(f"\n{'='*70}")
    print(f"Complete | Generations: {generation}")
    print(f"Best score: {best_score:.1f}")
    print(f"{'='*70}\n")

    out = Path(__file__).parent / "best_solution.py"
    out.write_text(best_program.code)
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
