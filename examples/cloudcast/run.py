#!/usr/bin/env python3
"""
AlphaEvolve on Cloudcast Broadcast Optimization Problem.

Uses per-config performance as behavioral dimensions for MAP-Elites diversity.
"""

import asyncio
import sys
import time
import json
import re
import tempfile
import os
from pathlib import Path
from concurrent.futures import BrokenExecutor

import numpy as np

# Add algoforge to path
ALGOFORGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ALGOFORGE_ROOT))

# Add cloudcast resources to path
CLOUDCAST_RESOURCES = ALGOFORGE_ROOT.parent / "ADRS-Leaderboard" / "problems" / "cloudcast" / "resources"
sys.path.insert(0, str(CLOUDCAST_RESOURCES))

from algoforge.core import Program, EvaluationResult
from algoforge.pool import CVTMAPElitesPool
from algoforge.llm import PromptBuilder, ProgramWithScore, OutputMode
from algoforge.behavior import BehaviorExtractor, FeatureVector
from algoforge.utils import ResilientProcessPool

from prompts import (
    PROBLEM_DESCRIPTION,
    FUNCTION_SIGNATURE,
    SEED_PROGRAM,
    SEED_INSPIRATIONS,
    DIVERSITY_SEED_PROMPT,
    META_ADVISOR_PROMPT,
)

import litellm
import logging

# Suppress LiteLLM debug spam
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
litellm.suppress_debug_info = True
litellm.set_verbose = False

# Register custom models
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

# Configuration names for per-config behavioral dimensions
CONFIG_NAMES = ["intra_aws", "intra_azure", "intra_gcp", "inter_agz", "inter_gaz2"]

# Scoring constants
LOWER_COST = 1199.00  # worst case (baseline)
UPPER_COST = 626.24   # best known


def validate_python_syntax(code: str) -> tuple[bool, str]:
    """Check if code is valid Python syntax. Returns (is_valid, error_message)."""
    try:
        compile(code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def quick_validate_code(code: str) -> tuple[bool, str]:
    """Quick validation to catch obvious errors before full evaluation."""
    if 'def search_algorithm' not in code:
        return False, "Missing search_algorithm function"

    if 'BroadCastTopology' not in code:
        return False, "Missing BroadCastTopology class"

    # Try to compile and exec with mocks
    try:
        import networkx as nx

        namespace = {
            'nx': nx,
            'networkx': nx,
            '__builtins__': __builtins__,
        }

        exec(code, namespace)

        if 'search_algorithm' not in namespace:
            return False, "search_algorithm not defined after exec"

        return True, ""
    except Exception as e:
        return False, f"Quick validation failed: {str(e)[:100]}"


def extract_and_validate_code(response: str) -> tuple[str | None, str]:
    """Extract and validate Python code from LLM response. Returns (code, error_msg)."""
    # Strip thinking tags
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
        if 'def search_algorithm' in match:
            candidates.append(match.strip())

    # Try raw extraction if no code blocks
    if not candidates and 'def search_algorithm' in response:
        lines = response.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if 'import ' in line or 'from ' in line or 'def search_' in line or 'class Broad' in line:
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
        is_valid, syntax_err = validate_python_syntax(candidate)
        if not is_valid:
            last_error = f"Syntax: {syntax_err}"
            continue

        is_valid, validation_err = quick_validate_code(candidate)
        if is_valid:
            return candidate, ""
        else:
            last_error = validation_err

    return None, last_error


def evaluate_code_in_process(code: str) -> dict:
    """Execute cloudcast code in a subprocess. Must be picklable."""
    import sys
    import json
    import tempfile
    import os
    from pathlib import Path

    # Ensure paths are set up in worker process
    algoforge_root = Path(__file__).resolve().parents[2]
    cloudcast_resources = algoforge_root.parent / "ADRS-Leaderboard" / "problems" / "cloudcast" / "resources"
    if str(cloudcast_resources) not in sys.path:
        sys.path.insert(0, str(cloudcast_resources))

    from simulator import BCSimulator
    from utils import make_nx_graph

    # Config files (ADRS-Leaderboard directory structure)
    config_dir = cloudcast_resources / "datasets" / "examples" / "config"
    cost_csv = cloudcast_resources / "datasets" / "profiles" / "cost.csv"
    throughput_csv = cloudcast_resources / "datasets" / "profiles" / "throughput.csv"

    config_files = {
        "intra_aws": config_dir / "intra_aws.json",
        "intra_azure": config_dir / "intra_azure.json",
        "intra_gcp": config_dir / "intra_gcp.json",
        "inter_agz": config_dir / "inter_agz.json",
        "inter_gaz2": config_dir / "inter_gaz2.json",
    }

    num_vms = 2

    try:
        # Execute code to get search_algorithm function
        namespace = {'__builtins__': __builtins__}
        exec(code, namespace)

        if 'search_algorithm' not in namespace:
            return {"error": "search_algorithm not defined"}

        search_algorithm = namespace['search_algorithm']
        BroadCastTopology = namespace.get('BroadCastTopology')

        # Track per-config costs
        per_config_costs = {}
        per_config_times = {}
        total_cost = 0.0
        total_time = 0.0
        successful = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                graph = make_nx_graph(
                    cost_path=str(cost_csv),
                    throughput_path=str(throughput_csv),
                    num_vms=num_vms,
                )

                for config_name, config_path in config_files.items():
                    try:
                        config = json.loads(config_path.read_text(encoding="utf-8"))

                        # Run search algorithm
                        bc_topology = search_algorithm(
                            config["source_node"],
                            config["dest_nodes"],
                            graph,
                            config["num_partitions"],
                        )

                        bc_topology.set_num_partitions(config["num_partitions"])

                        # Evaluate
                        simulator = BCSimulator(num_vms=num_vms, output_dir="evals")
                        transfer_time, cost = simulator.evaluate_path(bc_topology, config)

                        per_config_costs[config_name] = cost
                        per_config_times[config_name] = transfer_time
                        total_cost += cost
                        total_time += transfer_time
                        successful += 1

                    except Exception as e:
                        # If any config fails, mark it
                        per_config_costs[config_name] = LOWER_COST / 5  # worst case per config
                        per_config_times[config_name] = 999.0
                        return {"error": f"Config {config_name} failed: {str(e)[:100]}"}

            finally:
                os.chdir(original_cwd)

        if successful == 0:
            return {"error": "No configurations evaluated successfully"}

        # Compute score using leaderboard formula
        cost_clamped = max(min(total_cost, LOWER_COST), UPPER_COST)
        normalized_cost = (LOWER_COST - cost_clamped) / (LOWER_COST - UPPER_COST)
        score = normalized_cost * 100

        return {
            "score": score,
            "total_cost": total_cost,
            "total_time": total_time,
            "successful_configs": successful,
            "per_config_costs": per_config_costs,
            "per_config_times": per_config_times,
        }

    except Exception as e:
        return {"error": str(e)[:200]}


class CloudcastBehaviorExtractor(BehaviorExtractor):
    """
    Custom behavior extractor that uses per-config performance as behavioral dimensions.
    This enables MAP-Elites to maintain diversity across different network topologies.
    """

    def __init__(self):
        # Features: normalized cost for each config (lower = better, so we invert)
        self.features = [f"cost_{name}" for name in CONFIG_NAMES] + ["execution_time", "primary_score"]
        self.max_cost_per_config = LOWER_COST / 5  # ~240 per config baseline

    def extract(self, program: Program) -> FeatureVector:
        """Extract per-config performance as behavioral features."""
        metadata = program.metadata or {}
        values = {}

        # Per-config cost features (normalized: 0 = worst, 1 = best)
        per_config_costs = metadata.get("per_config_costs", {})
        for name in CONFIG_NAMES:
            cost = per_config_costs.get(name, self.max_cost_per_config)
            # Normalize: max_cost -> 0, 0 -> 1
            values[f"cost_{name}"] = max(0.0, 1.0 - cost / self.max_cost_per_config)

        # Execution time (normalized: slow -> 0, fast -> 1)
        exec_time = metadata.get("execution_time", 60.0)
        values["execution_time"] = max(0.0, 1.0 - exec_time / 60.0)

        # Primary score (normalized: 0 -> 0, 100 -> 1)
        score = metadata.get("primary_score", 0.0)
        values["primary_score"] = score / 100.0

        return FeatureVector(values)


# State tracking
STATE_FILE = Path(__file__).parent / "evolution_state.json"


def save_state(
    generation: int,
    pool: CVTMAPElitesPool,
    best_score: float,
    best_program: Program,
    total_cost_usd: float,
    extra_info: dict = None,
):
    """Save current evolution state to JSON file."""
    elites = []
    try:
        for cell_idx, elite in pool._elites.items():
            elite_info = {
                "cell_index": int(cell_idx),
                "score": float(elite.result.primary_score),
                "code": elite.program.code,
                "metadata": {k: float(v) if isinstance(v, (int, float)) else str(v)
                            for k, v in (elite.program.metadata or {}).items()
                            if not isinstance(v, dict)},
            }
            if elite.behavior:
                elite_info["behavior"] = {k: float(v) for k, v in elite.behavior.values.items()}
            elites.append(elite_info)
    except AttributeError:
        pass

    elites.sort(key=lambda x: x["score"], reverse=True)

    state = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generation": generation,
        "archive_size": pool.size(),
        "best_score": float(best_score),
        "total_cost_usd": float(total_cost_usd),
        "best_program": {
            "code": best_program.code,
            "metadata": {k: float(v) if isinstance(v, (int, float)) else str(v)
                        for k, v in (best_program.metadata or {}).items()
                        if not isinstance(v, dict)},
        },
        "archive": elites,
    }

    if extra_info:
        state.update(extra_info)

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
    """Format metrics data for the LLM to analyze."""
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
    """Use heavy LLM to generate strategic meta-advice from evolution metrics."""
    metrics_data = format_metrics_for_llm(metrics, previous_advice, best_score, top_solutions)
    prompt = META_ADVISOR_PROMPT.format(metrics_data=metrics_data)

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            timeout=60,
        )
        advice = response.choices[0].message.content.strip()
        cost = litellm.completion_cost(completion_response=response)
        return advice, cost
    except Exception as e:
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


def compute_score(output: dict) -> float:
    """Compute 0-100 score from evaluation output."""
    if output is None or "error" in output:
        return 0.0
    return output.get("score", 0.0)


def main():
    import random

    # Model configuration
    LIGHT_MODELS = [
        'openrouter/qwen/qwen3-coder-30b-a3b-instruct',
        'openrouter/google/gemini-2.5-flash-lite',
        'openrouter/deepseek/deepseek-v3.2',
    ]
    HEAVY_MODEL = 'openrouter/deepseek/deepseek-v3.2'

    n_workers = 8
    n_inspirations = 2

    # Use custom behavior extractor with per-config dimensions
    extractor = CloudcastBehaviorExtractor()

    # Single archive with deferred centroid initialization
    pool = CVTMAPElitesPool(
        behavior_extractor=extractor,
        n_centroids=50,
        subscore_keys=["execution_time"],
        defer_centroids=True,
    )

    # Evaluate seed program
    print(f"Evaluating seed program...", flush=True)

    seed_output = evaluate_code_in_process(SEED_PROGRAM)
    seed_score = compute_score(seed_output)
    seed_exec_time = seed_output.get("total_time", 60.0)
    seed_total_cost = seed_output.get("total_cost", LOWER_COST)
    print(f"  Seed: cost=${seed_total_cost:.2f}, score={seed_score:.1f}", flush=True)

    seed_metadata = {
        "execution_time": seed_exec_time,
        "primary_score": seed_score,
        "per_config_costs": seed_output.get("per_config_costs", {}),
    }
    seed_program = Program(code=SEED_PROGRAM, metadata=seed_metadata)
    best_score = seed_score
    best_program = seed_program

    sampler_names = pool.get_sampler_names()

    print(f"{'='*70}")
    print(f"AlphaEvolve - Cloudcast Broadcast Optimization")
    print(f"{'='*70}")
    print(f"  Baseline cost:        ${LOWER_COST:.2f}")
    print(f"  Best known:           ${UPPER_COST:.2f}")
    print(f"  Seed cost:            ${seed_total_cost:.2f} (score: {seed_score:.1f})")
    print(f"  Budget:               $5.00")
    print(f"  Behavior dimensions:  {', '.join(CONFIG_NAMES)} + time + score")
    print(f"  Samplers:             {', '.join(sampler_names)} (UCB runs 3x)")
    print(f"  Light models:         {', '.join(LIGHT_MODELS)}")
    print(f"  Heavy model:          {HEAVY_MODEL}")
    print(f"{'='*70}\n")

    generation = 0
    total_cost = 0.0

    executor = ResilientProcessPool(max_workers=n_workers, max_tasks_per_child=5)

    async def initialize_archive():
        """Generate diverse seeds with heavy model, then expand with light models."""
        nonlocal total_cost, best_score, best_program, seed_score, seed_output

        loop = asyncio.get_event_loop()

        # Phase 1: Generate diverse seeds using heavy model
        n_diverse_seeds = 5
        print(f"\n[Init Phase 1] Generating {n_diverse_seeds} diverse seeds with heavy model...", flush=True)

        diverse_seeds = [(SEED_PROGRAM, seed_score, seed_output.get("per_config_costs", {}))]

        for i in range(n_diverse_seeds):
            existing_seeds_text = "\n\n---\n\n".join([
                f"### Seed {j+1} (Score: {score:.1f}):\n```python\n{code}\n```"
                for j, (code, score, _) in enumerate(diverse_seeds)
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
                try:
                    eval_result = await loop.run_in_executor(executor.executor, evaluate_code_in_process, new_code)
                except BrokenExecutor:
                    executor._recreate_executor()
                    eval_result = {"error": "Pool crashed, recreated"}
                if "error" not in eval_result:
                    new_score = compute_score(eval_result)
                    diverse_seeds.append((new_code, new_score, eval_result.get("per_config_costs", {})))
                    print(f"  [Seed {i+1}] OK - score: {new_score:.1f}, tokens: {result['tokens']}", flush=True)
                else:
                    print(f"  [Seed {i+1}] EVAL FAIL: {eval_result['error'][:50]}", flush=True)
            else:
                print(f"  [Seed {i+1}] VALIDATION FAIL: {validation_error}", flush=True)

        print(f"[Init Phase 1] Generated {len(diverse_seeds)-1} new diverse seeds (total: {len(diverse_seeds)})", flush=True)

        # Phase 2: Generate variants using light models
        n_variants_per_seed = 20
        n_variants = n_variants_per_seed * len(diverse_seeds)
        print(f"\n[Init Phase 2] Generating {n_variants} variants ({n_variants_per_seed} per seed) with light models...", flush=True)

        prompts = []
        for seed_idx, (seed_code, s_score, _) in enumerate(diverse_seeds):
            seed_prog = Program(code=seed_code, metadata={"score": s_score})
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

        llm_tasks = [
            generate_for_island(i, prompts[i], LIGHT_MODELS[i % len(LIGHT_MODELS)], 0.8)
            for i in range(n_variants)
        ]
        results = await asyncio.gather(*llm_tasks)
        print(f"[Init Phase 2] All {n_variants} LLM calls complete", flush=True)

        candidates = []
        for res in results:
            if "error" in res:
                continue

            total_cost += res["cost"]
            idx = res["island"]
            tokens = res.get("tokens", 0)
            model = res.get("model", "unknown")

            new_code, validation_error = extract_and_validate_code(res["content"])

            if not new_code:
                continue

            candidates.append({"idx": idx, "code": new_code, "tokens": tokens, "model": model})

        print(f"[Init] Evaluating {len(candidates)} candidates...", flush=True)

        eval_map = {}
        completed = 0
        semaphore = asyncio.Semaphore(8)

        async def eval_candidate(idx, code):
            nonlocal completed
            async with semaphore:
                start = time.time()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(executor.executor, evaluate_code_in_process, code),
                        timeout=60
                    )
                    elapsed = time.time() - start
                    completed += 1
                    print(f"  [Init Eval] {completed}/{len(candidates)} Candidate {idx} done in {elapsed:.1f}s", flush=True)
                    return idx, result
                except asyncio.TimeoutError:
                    completed += 1
                    print(f"  [Init Eval] {completed}/{len(candidates)} Candidate {idx} TIMEOUT", flush=True)
                    return idx, {"error": "Timeout"}
                except BrokenExecutor:
                    executor._recreate_executor()
                    completed += 1
                    return idx, {"error": "Pool crashed, recreated"}
                except Exception as e:
                    completed += 1
                    return idx, {"error": str(e)}

        eval_tasks = [eval_candidate(c["idx"], c["code"]) for c in candidates]
        eval_results = await asyncio.gather(*eval_tasks)
        eval_map = {idx: res for idx, res in eval_results}

        print(f"[Init] Processing {len(candidates)} candidates...", flush=True)

        valid_programs = []
        behavior_vectors = []
        n_errors = 0
        for cand in candidates:
            output = eval_map.get(cand["idx"], {"error": "missing"})
            if "error" in output:
                n_errors += 1
                continue
            score = compute_score(output)
            execution_time = output.get("total_time", 60.0)
            valid_programs.append({
                "code": cand["code"],
                "score": score,
                "output": output,
                "execution_time": execution_time,
                "per_config_costs": output.get("per_config_costs", {}),
            })
            temp_prog = Program(code=cand["code"], metadata={
                "execution_time": execution_time,
                "primary_score": score,
                "per_config_costs": output.get("per_config_costs", {}),
            })
            behavior = extractor.extract(temp_prog)
            behavior_vectors.append(np.array([behavior[f] for f in extractor.features]))

        print(f"[Init] Valid programs: {len(valid_programs)}/{len(candidates)} ({n_errors} eval failures)", flush=True)

        # Add diverse seeds to valid_programs
        print(f"[Init] Adding {len(diverse_seeds)} diverse seeds to candidates...", flush=True)
        for seed_code, seed_score, per_config_costs in diverse_seeds:
            try:
                seed_eval = await loop.run_in_executor(executor.executor, evaluate_code_in_process, seed_code)
            except BrokenExecutor:
                executor._recreate_executor()
                seed_eval = {"error": "Pool crashed"}
            if "error" not in seed_eval:
                execution_time = seed_eval.get("total_time", 60.0)
                valid_programs.append({
                    "code": seed_code,
                    "score": seed_score,
                    "output": seed_eval,
                    "execution_time": execution_time,
                    "per_config_costs": seed_eval.get("per_config_costs", {}),
                })
                temp_prog = Program(code=seed_code, metadata={
                    "execution_time": execution_time,
                    "primary_score": seed_score,
                    "per_config_costs": seed_eval.get("per_config_costs", {}),
                })
                behavior = extractor.extract(temp_prog)
                behavior_vectors.append(np.array([behavior[f] for f in extractor.features]))

        print(f"[Init] Total valid programs: {len(valid_programs)}", flush=True)

        # Select top 50 by score
        valid_programs.sort(key=lambda x: x["score"], reverse=True)
        top_programs = valid_programs[:50]
        print(f"[Init] Selected top {len(top_programs)} programs by score", flush=True)

        # Build centroids from top programs
        top_behaviors = []
        for prog in top_programs:
            temp_prog = Program(code=prog["code"], metadata={
                "execution_time": prog["execution_time"],
                "primary_score": prog["score"],
                "per_config_costs": prog.get("per_config_costs", {}),
            })
            behavior = extractor.extract(temp_prog)
            top_behaviors.append(np.array([behavior[f] for f in extractor.features]))

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
                "per_config_costs": prog.get("per_config_costs", {}),
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

        save_state(
            generation=0,
            pool=pool,
            best_score=best_score,
            best_program=best_program,
            total_cost_usd=total_cost,
            extra_info={"phase": "initialization", "candidates_accepted": n_accepted},
        )
        print(f"[Init] State saved to {STATE_FILE}\n", flush=True)

    async def run_generation():
        nonlocal generation, best_score, best_program, total_cost

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

            # Every 10 generations, generate new meta-advice
            if generation > 1 and (generation - 1) % 10 == 0:
                top_solutions = []
                try:
                    elites = list(pool._elites.values())
                    elites.sort(key=lambda e: e.result.primary_score, reverse=True)
                    for elite in elites[:3]:
                        score = elite.result.primary_score
                        code = elite.program.code
                        top_solutions.append((score, code))
                except Exception:
                    pass

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

            # Build prompts for each sampler
            active_samplers = []
            for name in sampler_names:
                sampler = pool.get_sampler(name)
                if sampler.model_type == "heavy":
                    active_samplers.append((name, name, HEAVY_MODEL))
                elif name == "ucb":
                    for ucb_idx in range(3):
                        active_samplers.append((f"ucb_{ucb_idx}", "ucb", random.choice(LIGHT_MODELS)))
                else:
                    active_samplers.append((name, name, random.choice(LIGHT_MODELS)))

            prompts = []
            sampler_data = []
            for display_name, real_sampler, model in active_samplers:
                sample = pool.sample(real_sampler, n_parents=1 + n_inspirations)
                parents = [sample.parent] + sample.inspirations
                source_cell = sample.metadata.get("source_cell", 0)

                builder = PromptBuilder()
                builder.add_section("Problem", PROBLEM_DESCRIPTION, priority=10)
                builder.add_section("Signature", f"```python\n{FUNCTION_SIGNATURE}\n```", priority=20)
                builder.add_parents([ProgramWithScore(p, None) for p in parents], priority=30)
                builder.set_output_mode(OutputMode.FULL)

                if current_meta_advice:
                    builder.add_section("Meta-Advice", current_meta_advice, priority=100)

                prompts.append(builder.build())
                sampler_data.append({
                    "sampler": display_name,
                    "real_sampler": real_sampler,
                    "model": model,
                    "parents": parents,
                    "source_cell": source_cell,
                })

            n_samplers = len(active_samplers)
            print(f"[Gen {generation}] LLM calls for {n_samplers} samplers...", flush=True)
            llm_tasks = [
                generate_for_island(i, prompts[i], sampler_data[i]["model"], 0.8)
                for i in range(n_samplers)
            ]
            results = await asyncio.gather(*llm_tasks)
            print(f"[Gen {generation}] All LLM calls complete", flush=True)

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

            EVAL_TIMEOUT = 60  # 1 minute timeout for cloudcast (fast problem)
            print(f"[Gen {generation}] Evaluating {len(candidates)} candidates...", flush=True)
            eval_semaphore = asyncio.Semaphore(8)

            async def eval_with_timeout(idx, code):
                async with eval_semaphore:
                    start = time.time()
                    try:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(executor.executor, evaluate_code_in_process, code),
                            timeout=EVAL_TIMEOUT
                        )
                        elapsed = time.time() - start
                        print(f"  [Eval] Candidate {idx} done in {elapsed:.1f}s", flush=True)
                        return idx, result
                    except asyncio.TimeoutError:
                        print(f"  [Eval] Candidate {idx} TIMEOUT after {EVAL_TIMEOUT}s", flush=True)
                        return idx, {"error": f"Timeout after {EVAL_TIMEOUT}s"}
                    except BrokenExecutor:
                        executor._recreate_executor()
                        return idx, {"error": "Pool crashed, recreated"}
                    except Exception as e:
                        return idx, {"error": str(e)}

            eval_tasks = [eval_with_timeout(i, c["code"]) for i, c in enumerate(candidates)]
            indexed_results = await asyncio.gather(*eval_tasks)
            eval_results = [r for _, r in sorted(indexed_results, key=lambda x: x[0])]
            print(f"[Gen {generation}] All evaluations complete", flush=True)

            batch_time = time.time() - batch_start

            for cand, output in zip(candidates, eval_results):
                display_name = cand["sampler"]
                real_sampler = cand["real_sampler"]
                tokens = cand["tokens"]
                score = compute_score(output)
                exec_time = output.get("total_time", 60.0)

                child = Program(code=cand["code"], metadata={
                    "execution_time": exec_time,
                    "primary_score": score,
                    "per_config_costs": output.get("per_config_costs", {}),
                })
                eval_result = EvaluationResult(
                    program_id=child.id,
                    scores={'score': score},
                    is_valid="error" not in output,
                    error=output.get("error"),
                )

                period_metrics['total'] += 1

                if eval_result.is_valid:
                    accepted = pool.add(child, eval_result)
                    pool.update_sampler(real_sampler, cand["source_cell"], success=accepted, reward=score)
                    total_cost_val = output.get("total_cost", LOWER_COST)
                    status = "accepted" if accepted else "rejected"

                    if accepted:
                        period_metrics['acceptances'] += 1
                    else:
                        period_metrics['rejections'] += 1

                    if score > best_score:
                        best_score, best_program = score, child
                        status = "NEW BEST ★"
                        period_metrics['new_bests'] += 1

                    print(f"[Gen {generation}] {display_name:20s} {status:10s} | cost: ${total_cost_val:6.2f} | score: {score:5.1f} | best: {best_score:5.1f} | {tokens}tok | ${total_cost:.3f}", flush=True)
                else:
                    pool.update_sampler(real_sampler, cand["source_cell"], success=False)
                    err = eval_result.error[:30] if eval_result.error else "unknown"

                    if eval_result.error and "timeout" in eval_result.error.lower():
                        period_metrics['timeouts'] += 1
                    else:
                        period_metrics['errors'] += 1
                        if eval_result.error:
                            err_msg = eval_result.error[:100].strip()
                            period_metrics['error_messages'].add(err_msg)

                    print(f"[Gen {generation}] {display_name:20s} INVALID    | {err}...", flush=True)

            pool.on_generation_complete()
            print(f"    [Batch {generation} done in {batch_time:.1f}s | {n_samplers} samplers, archive: {pool.size()}]", flush=True)

            if generation % 25 == 0:
                print(f"{'─'*70}\n[MILESTONE] Best score: {best_score:.1f}\n{'─'*70}", flush=True)

            save_state(
                generation=generation,
                pool=pool,
                best_score=best_score,
                best_program=best_program,
                total_cost_usd=total_cost,
                extra_info={
                    "phase": "evolution",
                    "batch_time_s": batch_time,
                    "n_samplers": n_samplers,
                },
            )

        executor.shutdown(wait=False)

    async def main_async():
        await initialize_archive()
        await run_generation()

    asyncio.run(main_async())

    print(f"\n{'='*70}")
    print(f"Complete | Generations: {generation}")
    print(f"Best score: {best_score:.1f}")
    print(f"{'='*70}\n")

    out = Path(__file__).parent / "best_solution.py"
    out.write_text(best_program.code)
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
