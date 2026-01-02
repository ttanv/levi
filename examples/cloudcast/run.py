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
from contextlib import redirect_stdout
from pathlib import Path

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
        import math
        import time as time_module
        import random
        import collections
        import heapq
        import numpy as np
        from typing import Dict, List, Set, Tuple, Any

        # Mock BroadCastTopology class for validation
        class MockBroadCastTopology:
            def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict = None):
                self.src = src
                self.dsts = dsts
                self.num_partitions = num_partitions
                self.paths = paths or {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

            def set_num_partitions(self, num_partitions: int):
                self.num_partitions = num_partitions

            def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
                partition = str(partition)
                self.paths[dst][partition] = paths

            def append_dst_partition_path(self, dst: str, partition: int, path: List):
                partition = str(partition)
                if self.paths[dst][partition] is None:
                    self.paths[dst][partition] = []
                self.paths[dst][partition].append(path)

        # Mock broadcast module
        class MockBroadcastModule:
            BroadCastTopology = MockBroadCastTopology

        import sys
        sys.modules['broadcast'] = MockBroadcastModule()

        namespace = {
            '__builtins__': __builtins__,
            # Core cloudcast classes
            'BroadCastTopology': MockBroadCastTopology,
            # NetworkX
            'nx': nx,
            'networkx': nx,
            # Common modules
            'math': math,
            'time': time_module,
            'random': random,
            'collections': collections,
            'heapq': heapq,
            'np': np,
            'numpy': np,
            # Typing
            'Dict': Dict,
            'List': List,
            'Set': Set,
            'Tuple': Tuple,
            'Any': Any,
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


def compute_config_score(cost: float, baseline: float, optimal: float) -> float:
    """Compute 0-1 normalized score for a single config (like workload scores)."""
    if cost >= baseline:
        return 0.0
    if cost <= optimal:
        return 1.0
    return (baseline - cost) / (baseline - optimal)


def evaluate_code_in_process(code: str) -> dict:
    """Execute cloudcast code in a subprocess. Must be picklable."""
    import sys
    import json
    import tempfile
    import os
    import math
    import time as time_module
    import random
    import collections
    import heapq
    from pathlib import Path
    from typing import Dict, List, Set, Tuple, Any

    # Ensure paths are set up in worker process
    algoforge_root = Path(__file__).resolve().parents[2]
    cloudcast_resources = algoforge_root.parent / "ADRS-Leaderboard" / "problems" / "cloudcast" / "resources"
    if str(cloudcast_resources) not in sys.path:
        sys.path.insert(0, str(cloudcast_resources))

    from simulator import BCSimulator
    from utils import make_nx_graph
    from broadcast import BroadCastTopology
    import networkx as nx
    import numpy as np

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

    # Per-config baselines and optimals for score calculation
    per_config_baseline = LOWER_COST / 5  # ~240 per config
    per_config_optimal = UPPER_COST / 5   # ~125 per config

    try:
        # Execute code to get search_algorithm function
        # Include common modules in namespace so LLM code doesn't need to import
        namespace = {
            '__builtins__': __builtins__,
            # Core cloudcast classes
            'BroadCastTopology': BroadCastTopology,
            # NetworkX
            'nx': nx,
            'networkx': nx,
            # Common modules (like txn_scheduling)
            'math': math,
            'time': time_module,
            'random': random,
            'collections': collections,
            'heapq': heapq,
            'np': np,
            'numpy': np,
            # Typing
            'Dict': Dict,
            'List': List,
            'Set': Set,
            'Tuple': Tuple,
            'Any': Any,
        }
        exec(code, namespace)

        if 'search_algorithm' not in namespace:
            return {"error": "search_algorithm not defined"}

        search_algorithm = namespace['search_algorithm']

        # Track per-config costs and scores
        per_config_costs = {}
        per_config_times = {}
        per_config_scores = {}
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

                        # Evaluate (suppress simulator's verbose output)
                        simulator = BCSimulator(num_vms=num_vms, output_dir="evals")
                        with open(os.devnull, 'w') as devnull, redirect_stdout(devnull):
                            transfer_time, cost = simulator.evaluate_path(bc_topology, config)

                        per_config_costs[config_name] = cost
                        per_config_times[config_name] = transfer_time
                        # Compute 0-1 normalized score for this config (like workload scores)
                        per_config_scores[config_name] = compute_config_score(
                            cost, per_config_baseline, per_config_optimal
                        )
                        total_cost += cost
                        total_time += transfer_time
                        successful += 1

                    except Exception as e:
                        # If any config fails, mark it
                        per_config_costs[config_name] = per_config_baseline  # worst case
                        per_config_times[config_name] = 999.0
                        per_config_scores[config_name] = 0.0  # worst score
                        return {"error": f"Config {config_name} failed: {str(e)[:100]}"}

            finally:
                os.chdir(original_cwd)

        if successful == 0:
            return {"error": "No configurations evaluated successfully"}

        # Validate solution actually transferred data (catch buggy solutions)
        # Some LLM-generated code has bugs (e.g., set_num_partitions clears paths)
        # that result in $0 cost and -inf time, which would incorrectly score 100
        if total_cost == 0:
            return {"error": "Invalid solution: zero cost (no data transferred)"}
        if not math.isfinite(total_time) or total_time <= 0:
            return {"error": f"Invalid solution: invalid transfer time ({total_time})"}

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
            # Per-config scores (0-1 normalized) for behavior extraction
            "intra_aws_score": per_config_scores.get("intra_aws", 0.0),
            "intra_azure_score": per_config_scores.get("intra_azure", 0.0),
            "intra_gcp_score": per_config_scores.get("intra_gcp", 0.0),
            "inter_agz_score": per_config_scores.get("inter_agz", 0.0),
            "inter_gaz2_score": per_config_scores.get("inter_gaz2", 0.0),
        }

    except Exception as e:
        return {"error": str(e)[:200]}


class CloudcastBehaviorExtractor(BehaviorExtractor):
    """
    Custom behavior extractor for cloudcast with z-score normalization.

    Uses per-config scores as behavioral dimensions (similar to workload_1/2/3 in txn_scheduling).
    Removes execution_time as a behavioral dimension per user request.

    Features (all normalized to ~[0, 1] via sigmoid of z-score):
    - loop_count: Number of for/while loops
    - branch_count: Number of if/elif/else branches
    - math_operators: Count of math ops
    - intra_aws_score: Performance on intra_aws config
    - intra_azure_score: Performance on intra_azure config
    - intra_gcp_score: Performance on intra_gcp config
    - inter_agz_score: Performance on inter_agz config
    - inter_gaz2_score: Performance on inter_gaz2 config
    """

    def __init__(self):
        self.features = [
            'loop_count',
            'branch_count',
            'math_operators',
            'intra_aws_score',
            'intra_azure_score',
            'intra_gcp_score',
            'inter_agz_score',
            'inter_gaz2_score',
        ]

        # All features use z-score normalization (Welford's online algorithm)
        self._zscore_features = self.features.copy()
        self._count = {f: 0 for f in self._zscore_features}
        self._mean = {f: 0.0 for f in self._zscore_features}
        self._M2 = {f: 0.0 for f in self._zscore_features}  # Sum of squared differences

        # Per-config baselines for score calculation
        self.per_config_baseline = LOWER_COST / 5  # ~240 per config
        self.per_config_optimal = UPPER_COST / 5   # ~125 per config

    def _update_stats(self, feature: str, value: float):
        """Update running mean and variance using Welford's algorithm."""
        self._count[feature] += 1
        delta = value - self._mean[feature]
        self._mean[feature] += delta / self._count[feature]
        delta2 = value - self._mean[feature]
        self._M2[feature] += delta * delta2

    def _get_std(self, feature: str) -> float:
        """Get current standard deviation for a feature."""
        if self._count[feature] < 2:
            return 1.0  # Avoid division by zero
        variance = self._M2[feature] / (self._count[feature] - 1)
        return max(np.sqrt(variance), 0.1)  # Min std of 0.1 to avoid extreme z-scores

    def _zscore_to_01(self, z: float) -> float:
        """Convert z-score to [0, 1] using sigmoid."""
        z = max(-10, min(10, z))  # Clamp to avoid overflow
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_config_score(self, cost: float) -> float:
        """Compute 0-1 normalized score for a single config (like workload scores)."""
        if cost >= self.per_config_baseline:
            return 0.0
        if cost <= self.per_config_optimal:
            return 1.0
        return (self.per_config_baseline - cost) / (self.per_config_baseline - self.per_config_optimal)

    def extract(self, program: Program) -> FeatureVector:
        """Extract behavioral features from a program."""
        import ast

        code = program.code
        metadata = program.metadata or {}
        values = {}

        # Parse AST for static features
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return FeatureVector({f: 0.5 for f in self.features})

        # === Static features (from code analysis) ===

        # Count loops
        loop_count = sum(1 for node in ast.walk(tree)
                        if isinstance(node, (ast.For, ast.While)))

        # Count branches
        branch_count = sum(1 for node in ast.walk(tree)
                          if isinstance(node, ast.If))

        # Count math operators
        math_ops = sum(1 for node in ast.walk(tree)
                      if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.AugAssign)))

        # === Dynamic features (per-config scores from evaluation) ===
        intra_aws = metadata.get('intra_aws_score', 0.0)
        intra_azure = metadata.get('intra_azure_score', 0.0)
        intra_gcp = metadata.get('intra_gcp_score', 0.0)
        inter_agz = metadata.get('inter_agz_score', 0.0)
        inter_gaz2 = metadata.get('inter_gaz2_score', 0.0)

        # Collect all raw values
        raw_values = {
            'loop_count': float(loop_count),
            'branch_count': float(branch_count),
            'math_operators': float(math_ops),
            'intra_aws_score': float(intra_aws),
            'intra_azure_score': float(intra_azure),
            'intra_gcp_score': float(intra_gcp),
            'inter_agz_score': float(inter_agz),
            'inter_gaz2_score': float(inter_gaz2),
        }

        # Update running statistics and apply z-score normalization with sigmoid
        for feature in self._zscore_features:
            self._update_stats(feature, raw_values[feature])
            z = (raw_values[feature] - self._mean[feature]) / self._get_std(feature)
            values[feature] = self._zscore_to_01(z)

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

    # Retry up to 3 times for transient failures
    last_error = None
    for attempt in range(3):
        try:
            # Build call kwargs
            call_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 800,  # ~500 words max
                "timeout": 180,  # 3 minutes for meta-advice
            }

            # Enable reasoning for DeepSeek models
            if model and "deepseek" in model.lower():
                call_kwargs["reasoning"] = {"enabled": True}

            response = await litellm.acompletion(**call_kwargs)
            advice = response.choices[0].message.content.strip()
            cost = litellm.completion_cost(completion_response=response)
            return advice, cost
        except Exception as e:
            last_error = e
            if attempt < 2:
                print(f"[Meta-Advice] Attempt {attempt+1} failed: {str(e)[:50]}, retrying...", flush=True)
                await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
            continue

    # Fallback to simple formatted advice if all retries fail
    fallback = f"(Meta-advice generation failed after 3 attempts: {str(last_error)[:50]})\n\n"
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
    # Note: removed subscore_keys=["execution_time"] per user request
    pool = CVTMAPElitesPool(
        behavior_extractor=extractor,
        n_centroids=100,
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
        # Per-config scores for behavior extraction
        "intra_aws_score": seed_output.get("intra_aws_score", 0.0),
        "intra_azure_score": seed_output.get("intra_azure_score", 0.0),
        "intra_gcp_score": seed_output.get("intra_gcp_score", 0.0),
        "inter_agz_score": seed_output.get("inter_agz_score", 0.0),
        "inter_gaz2_score": seed_output.get("inter_gaz2_score", 0.0),
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
    print(f"  Per-config scores:    aws={seed_output.get('intra_aws_score', 0):.2f}, azure={seed_output.get('intra_azure_score', 0):.2f}, gcp={seed_output.get('intra_gcp_score', 0):.2f}, agz={seed_output.get('inter_agz_score', 0):.2f}, gaz2={seed_output.get('inter_gaz2_score', 0):.2f}")
    print(f"  Budget:               $5.00")
    print(f"  Behavior dimensions:  loop_count, branch_count, math_operators + 5 config scores")
    print(f"  Samplers:             {', '.join(sampler_names)} (UCB runs 3x)")
    print(f"  Light models:         {', '.join(LIGHT_MODELS)}")
    print(f"  Heavy model:          {HEAVY_MODEL}")
    print(f"{'='*70}\n")

    generation = 0
    total_cost = 0.0

    executor = ResilientProcessPool(max_workers=n_workers)

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
                    eval_result = await executor.run(evaluate_code_in_process, new_code, timeout=120)
                except Exception as e:
                    eval_result = {"error": str(e)}
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
                    result = await executor.run(evaluate_code_in_process, code, timeout=60)
                    elapsed = time.time() - start
                    completed += 1
                    print(f"  [Init Eval] {completed}/{len(candidates)} Candidate {idx} done in {elapsed:.1f}s", flush=True)
                    return idx, result
                except TimeoutError:
                    completed += 1
                    print(f"  [Init Eval] {completed}/{len(candidates)} Candidate {idx} TIMEOUT", flush=True)
                    return idx, {"error": "Timeout"}
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
                # Per-config scores for behavior extraction
                "intra_aws_score": output.get("intra_aws_score", 0.0),
                "intra_azure_score": output.get("intra_azure_score", 0.0),
                "intra_gcp_score": output.get("intra_gcp_score", 0.0),
                "inter_agz_score": output.get("inter_agz_score", 0.0),
                "inter_gaz2_score": output.get("inter_gaz2_score", 0.0),
            })
            temp_prog = Program(code=cand["code"], metadata={
                "execution_time": execution_time,
                "primary_score": score,
                "per_config_costs": output.get("per_config_costs", {}),
                "intra_aws_score": output.get("intra_aws_score", 0.0),
                "intra_azure_score": output.get("intra_azure_score", 0.0),
                "intra_gcp_score": output.get("intra_gcp_score", 0.0),
                "inter_agz_score": output.get("inter_agz_score", 0.0),
                "inter_gaz2_score": output.get("inter_gaz2_score", 0.0),
            })
            behavior = extractor.extract(temp_prog)
            behavior_vectors.append(np.array([behavior[f] for f in extractor.features]))

        print(f"[Init] Valid programs: {len(valid_programs)}/{len(candidates)} ({n_errors} eval failures)", flush=True)

        # Add diverse seeds to valid_programs
        print(f"[Init] Adding {len(diverse_seeds)} diverse seeds to candidates...", flush=True)
        for seed_code, seed_score, per_config_costs in diverse_seeds:
            try:
                seed_eval = await executor.run(evaluate_code_in_process, seed_code, timeout=120)
            except Exception as e:
                seed_eval = {"error": str(e)}
            if "error" not in seed_eval:
                execution_time = seed_eval.get("total_time", 60.0)
                valid_programs.append({
                    "code": seed_code,
                    "score": seed_score,
                    "output": seed_eval,
                    "execution_time": execution_time,
                    "per_config_costs": seed_eval.get("per_config_costs", {}),
                    "intra_aws_score": seed_eval.get("intra_aws_score", 0.0),
                    "intra_azure_score": seed_eval.get("intra_azure_score", 0.0),
                    "intra_gcp_score": seed_eval.get("intra_gcp_score", 0.0),
                    "inter_agz_score": seed_eval.get("inter_agz_score", 0.0),
                    "inter_gaz2_score": seed_eval.get("inter_gaz2_score", 0.0),
                })
                temp_prog = Program(code=seed_code, metadata={
                    "execution_time": execution_time,
                    "primary_score": seed_score,
                    "per_config_costs": seed_eval.get("per_config_costs", {}),
                    "intra_aws_score": seed_eval.get("intra_aws_score", 0.0),
                    "intra_azure_score": seed_eval.get("intra_azure_score", 0.0),
                    "intra_gcp_score": seed_eval.get("intra_gcp_score", 0.0),
                    "inter_agz_score": seed_eval.get("inter_agz_score", 0.0),
                    "inter_gaz2_score": seed_eval.get("inter_gaz2_score", 0.0),
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
                "intra_aws_score": prog.get("intra_aws_score", 0.0),
                "intra_azure_score": prog.get("intra_azure_score", 0.0),
                "intra_gcp_score": prog.get("intra_gcp_score", 0.0),
                "inter_agz_score": prog.get("inter_agz_score", 0.0),
                "inter_gaz2_score": prog.get("inter_gaz2_score", 0.0),
            })
            behavior = extractor.extract(temp_prog)
            top_behaviors.append(np.array([behavior[f] for f in extractor.features]))

        print(f"[Init] Building centroids from {len(top_behaviors)} behavior vectors...", flush=True)
        n_centroids = pool.set_centroids_from_data(
            top_behaviors,
            percentile_low=5.0,
            percentile_high=95.0,
            n_centroids=100,
        )
        print(f"[Init] Built {n_centroids} centroids", flush=True)

        # Add top programs to archive
        n_accepted = 0
        for prog in top_programs:
            child = Program(code=prog["code"], metadata={
                "execution_time": prog["execution_time"],
                "primary_score": prog["score"],
                "per_config_costs": prog.get("per_config_costs", {}),
                "intra_aws_score": prog.get("intra_aws_score", 0.0),
                "intra_azure_score": prog.get("intra_azure_score", 0.0),
                "intra_gcp_score": prog.get("intra_gcp_score", 0.0),
                "inter_agz_score": prog.get("inter_agz_score", 0.0),
                "inter_gaz2_score": prog.get("inter_gaz2_score", 0.0),
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

    async def run_pipeline():
        """
        Producer-consumer pipeline for async LLM sampling and evaluation.

        Architecture (same as txn_scheduling):
        - Sampler Queue: Yields (display_name, real_sampler, model) in fair distribution
        - LLM Producers: Pull from sampler queue, generate code, push to eval queue
        - Eval Consumers: Pull from eval queue, evaluate in process pool, push to result queue
        - Result Processor: Update archive, track metrics, trigger meta-advice every 50 evals
        """
        nonlocal generation, best_score, best_program, total_cost

        print(f"[Pipeline] Starting async evolution pipeline...", flush=True)

        # Configuration
        N_LLM_WORKERS = 4       # Concurrent async LLM calls (I/O bound)
        N_EVAL_PROCESSES = 4    # Process pool size (CPU bound)
        EVAL_TIMEOUT = 60       # 1 minute timeout for cloudcast (fast problem)
        META_ADVICE_INTERVAL = 50  # Generate meta-advice every N evals
        BUDGET_USD = 5.0

        # Queues for pipeline
        sampler_queue = asyncio.Queue()      # Sampler configs to process
        eval_queue = asyncio.Queue()         # Candidates awaiting evaluation
        result_queue = asyncio.Queue()       # Evaluated results

        # Semaphore to limit total in-flight work
        pipeline_capacity = asyncio.Semaphore(N_EVAL_PROCESSES + 4)

        # Shared state with lock
        state_lock = asyncio.Lock()
        state = {
            'total_cost': total_cost,
            'eval_count': 0,
            'best_score': best_score,
            'best_program': best_program,
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
        }

        # Stop signal
        stop_event = asyncio.Event()

        def get_sampler_cycle():
            """Generate one cycle of sampler configs with fair distribution."""
            configs = []
            for name in sampler_names:
                sampler = pool.get_sampler(name)
                if sampler.model_type == "heavy":
                    configs.append((name, name, HEAVY_MODEL))
                elif name == "ucb":
                    for ucb_idx in range(3):
                        configs.append((f"ucb_{ucb_idx}", "ucb", random.choice(LIGHT_MODELS)))
                else:
                    configs.append((name, name, random.choice(LIGHT_MODELS)))
            return configs

        async def sampler_feeder():
            """Continuously feed sampler configs to the queue."""
            while not stop_event.is_set():
                configs = get_sampler_cycle()
                for config in configs:
                    if stop_event.is_set():
                        break
                    await sampler_queue.put(config)
                await asyncio.sleep(0.1)

        async def llm_producer(worker_id: int):
            """Pull sampler config, generate LLM response, push to eval queue."""
            while not stop_event.is_set():
                slot_acquired = False
                slot_transferred = False
                try:
                    # Get next sampler config
                    try:
                        display_name, real_sampler, model = await asyncio.wait_for(
                            sampler_queue.get(), timeout=2.0
                        )
                    except asyncio.TimeoutError:
                        continue

                    # Acquire pipeline slot BEFORE LLM call
                    await pipeline_capacity.acquire()
                    slot_acquired = True

                    # Check budget
                    async with state_lock:
                        if state['total_cost'] >= BUDGET_USD:
                            stop_event.set()
                            break
                        state['llm_in_flight'] += 1
                        current_meta_advice = state['current_meta_advice']

                    # Sample from pool
                    sample = pool.sample(real_sampler, n_parents=1 + n_inspirations)
                    inspirations = [p for p in sample.inspirations if random.random() < 0.8]
                    parents = [sample.parent] + inspirations
                    source_cell = sample.metadata.get("source_cell", 0)

                    # Build prompt
                    builder = PromptBuilder()
                    builder.add_section("Problem", PROBLEM_DESCRIPTION, priority=10)
                    builder.add_section("Signature", f"```python\n{FUNCTION_SIGNATURE}\n```", priority=20)
                    builder.add_parents([ProgramWithScore(p, None) for p in parents], priority=30)
                    builder.set_output_mode(OutputMode.FULL)

                    # Meta-advice injection (80% probability)
                    if current_meta_advice and random.random() < 0.8:
                        builder.add_section("Meta-Advice", current_meta_advice, priority=100)

                    prompt = builder.build()

                    # Generate
                    result = await generate_for_island(worker_id, prompt, model, 0.8)

                    async with state_lock:
                        state['llm_in_flight'] -= 1

                    if "error" in result:
                        print(f"[LLM-{worker_id}] {display_name} ERROR: {result['error'][:50]}", flush=True)
                        continue

                    # Track cost
                    async with state_lock:
                        state['total_cost'] += result["cost"]
                        if state['total_cost'] >= BUDGET_USD:
                            stop_event.set()

                    # Extract code
                    new_code, validation_error = extract_and_validate_code(result["content"])
                    if not new_code:
                        print(f"[LLM-{worker_id}] {display_name} VALIDATION FAIL: {validation_error}", flush=True)
                        continue

                    # Push to eval queue
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
                        slot_transferred = True
                    except asyncio.CancelledError:
                        slot_transferred = True
                        async with state_lock:
                            if state['llm_in_flight'] > 0:
                                state['llm_in_flight'] -= 1
                        raise

                except asyncio.CancelledError:
                    async with state_lock:
                        if state['llm_in_flight'] > 0:
                            state['llm_in_flight'] -= 1
                    raise
                except Exception as e:
                    print(f"[LLM-{worker_id}] Unexpected error: {e}", flush=True)
                    async with state_lock:
                        if state['llm_in_flight'] > 0:
                            state['llm_in_flight'] -= 1
                finally:
                    if slot_acquired and not slot_transferred:
                        pipeline_capacity.release()

        async def eval_dispatcher():
            """Dispatch evaluations and push results to result queue."""
            pending_evals = set()

            async def run_eval(candidate):
                incremented = False
                output = None
                elapsed = 0
                start = time.time()
                try:
                    async with state_lock:
                        state['eval_in_flight'] += 1
                        incremented = True

                    try:
                        output = await executor.run(evaluate_code_in_process, candidate["code"], timeout=EVAL_TIMEOUT)
                        elapsed = time.time() - start
                    except TimeoutError:
                        output = {"error": f"Timeout after {EVAL_TIMEOUT}s"}
                        elapsed = EVAL_TIMEOUT
                    except asyncio.CancelledError:
                        output = {"error": "Evaluation cancelled"}
                        elapsed = time.time() - start
                        raise
                    except Exception as e:
                        output = {"error": str(e)}
                        elapsed = time.time() - start

                    await result_queue.put({
                        "candidate": candidate,
                        "output": output,
                        "elapsed": elapsed,
                    })
                except asyncio.CancelledError:
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
                    print(f"[Eval] Unexpected error in run_eval: {e}", flush=True)
                finally:
                    if incremented:
                        async with state_lock:
                            state['eval_in_flight'] -= 1
                    pipeline_capacity.release()

            while not stop_event.is_set() or not eval_queue.empty():
                try:
                    candidate = await asyncio.wait_for(eval_queue.get(), timeout=2.0)
                    task = asyncio.create_task(run_eval(candidate))
                    pending_evals.add(task)
                    task.add_done_callback(pending_evals.discard)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"[EvalDispatch] Unexpected error: {e}", flush=True)

            # Wait for pending evals
            if pending_evals:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending_evals, return_exceptions=True),
                        timeout=EVAL_TIMEOUT + 10
                    )
                except asyncio.TimeoutError:
                    print(f"[EvalDispatch] Pending evals didn't complete in time", flush=True)

        async def result_processor():
            """Process evaluation results, update archive, trigger meta-advice."""
            nonlocal generation, best_score, best_program, total_cost

            last_save_time = time.time()

            while not stop_event.is_set() or not result_queue.empty():
                try:
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

                    score = compute_score(output)
                    exec_time = output.get("total_time", 60.0)

                    child = Program(code=candidate["code"], metadata={
                        "execution_time": exec_time,
                        "primary_score": score,
                        "per_config_costs": output.get("per_config_costs", {}),
                        "intra_aws_score": output.get("intra_aws_score", 0.0),
                        "intra_azure_score": output.get("intra_azure_score", 0.0),
                        "intra_gcp_score": output.get("intra_gcp_score", 0.0),
                        "inter_agz_score": output.get("inter_agz_score", 0.0),
                        "inter_gaz2_score": output.get("inter_gaz2_score", 0.0),
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
                        total_cost_val = output.get("total_cost", LOWER_COST)

                        async with state_lock:
                            if accepted:
                                state['period_metrics']['acceptances'] += 1
                            else:
                                state['period_metrics']['rejections'] += 1

                            status = "accepted" if accepted else "rejected"
                            if score > state['best_score']:
                                state['best_score'] = score
                                state['best_program'] = child
                                best_score = score
                                best_program = child
                                status = "NEW BEST ★"
                                state['period_metrics']['new_bests'] += 1

                        print(f"[Eval #{eval_count:4d}] {display_name:20s} {status:10s} | cost: ${total_cost_val:6.2f} | score: {score:5.1f} | best: {best_score:5.1f} | {tokens}tok | ${current_cost:.3f}", flush=True)
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

                    # Save state periodically
                    if time.time() - last_save_time > 30:
                        async with state_lock:
                            save_state(
                                generation=eval_count,
                                pool=pool,
                                best_score=state['best_score'],
                                best_program=state['best_program'],
                                total_cost_usd=state['total_cost'],
                                extra_info={
                                    "phase": "pipeline",
                                    "eval_count": eval_count,
                                    "llm_in_flight": state['llm_in_flight'],
                                    "eval_in_flight": state['eval_in_flight'],
                                },
                            )
                        last_save_time = time.time()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"[ResultProc] Unexpected error: {e}", flush=True)

            # Final state update
            async with state_lock:
                total_cost = state['total_cost']
                best_score = state['best_score']
                best_program = state['best_program']
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
                        'total': 0, 'timeouts': 0, 'errors': 0,
                        'rejections': 0, 'acceptances': 0, 'new_bests': 0,
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
                          f"LLM: {state['llm_in_flight']} | Eval: {state['eval_in_flight']} | "
                          f"Archive: {pool.size()} | Best: {state['best_score']:.1f}\n", flush=True)

        # Start all workers
        print(f"[Pipeline] Starting {N_LLM_WORKERS} LLM workers, {N_EVAL_PROCESSES}-process eval pool...", flush=True)

        feeder_task = asyncio.create_task(sampler_feeder())
        llm_tasks = [asyncio.create_task(llm_producer(i)) for i in range(N_LLM_WORKERS)]
        eval_task = asyncio.create_task(eval_dispatcher())
        processor_task = asyncio.create_task(result_processor())
        monitor_task = asyncio.create_task(status_monitor())

        # Wait for budget exhaustion
        while not stop_event.is_set():
            await asyncio.sleep(1.0)
            async with state_lock:
                if state['total_cost'] >= BUDGET_USD:
                    stop_event.set()

        print(f"\n[Pipeline] Budget exhausted, draining queues...", flush=True)

        # Cancel feeder
        feeder_task.cancel()
        try:
            await feeder_task
        except asyncio.CancelledError:
            pass

        # Cancel LLM workers
        for task in llm_tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*llm_tasks, return_exceptions=True),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            print("[Pipeline] LLM workers didn't cancel in time", flush=True)

        # Wait for eval dispatcher
        await eval_task

        # Wait for result processor
        try:
            await asyncio.wait_for(processor_task, timeout=30.0)
        except asyncio.TimeoutError:
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

        # Final state
        async with state_lock:
            total_cost = state['total_cost']
            best_score = state['best_score']
            best_program = state['best_program']
            generation = state['eval_count']

        # Final save
        save_state(
            generation=generation,
            pool=pool,
            best_score=best_score,
            best_program=best_program,
            total_cost_usd=total_cost,
            extra_info={"phase": "pipeline_complete"},
        )

        executor.shutdown()
        print(f"[Pipeline] Complete. Total evals: {generation}", flush=True)

    async def main_async():
        await initialize_archive()
        await run_pipeline()

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
