"""
Cloudcast Broadcast Optimization problem definition.

Contains prompts, seed program, and scoring function for `levi.evolve_code`.
"""

import collections
import heapq
import json
import math
import os
import random
import sys
import tempfile
import time as time_module
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np

PROBLEM_DESCRIPTION = """
# Cloudcast Broadcast Optimization

## Problem
Optimize broadcast topology for multi-cloud data distribution. Find optimal paths from a
source to multiple destinations across AWS, Azure, and GCP to minimize transfer cost and time.

## Key Concepts
- Graph G has edge attributes: `cost` ($/GB) and `throughput` (Gbps)
- BroadCastTopology stores paths for each (destination, partition) pair
- Data is partitioned into `num_partitions` chunks that can take different paths
- Paths from source must cover all destinations for all partitions

## Objective
Minimize total transfer cost ($/GB) across 5 network configurations:
- intra_aws, intra_azure, intra_gcp (single cloud)
- inter_agz, inter_gaz2 (cross-cloud)

## Scoring (0-100)
```
LOWER_COST = 1199.00  # worst case
UPPER_COST = 626.24   # best known
cost_clamped = max(min(total_cost, LOWER_COST), UPPER_COST)
normalized_cost = (LOWER_COST - cost_clamped) / (LOWER_COST - UPPER_COST)
score = normalized_cost * 100
```

## APIs
- `G.nodes` - All nodes (cloud regions)
- `G.edges(data=True)` - All edges with attributes
- `G[src][dst]['cost']` - Cost per GB for edge
- `G[src][dst]['throughput']` - Throughput in Gbps
- `nx.dijkstra_path(G, src, dst, weight='cost')` - Shortest path by cost
- `BroadCastTopology(src, dsts, num_partitions)` - Create topology
- `bc_topology.append_dst_partition_path(dst, partition, [src, tgt, edge_data])` - Add path segment

## CRITICAL CONSTRAINTS
- All destinations must be reachable for all partitions
- Paths must use valid edges in graph G
- Algorithm should run quickly (under 10 seconds total)
"""

FUNCTION_SIGNATURE = """
import networkx as nx
from typing import List

def search_algorithm(src: str, dsts: List[str], G: nx.DiGraph, num_partitions: int):
    '''
    Find optimal broadcast topology from src to all destinations.

    Args:
        src: Source node (cloud region)
        dsts: List of destination nodes
        G: NetworkX DiGraph with 'cost' and 'throughput' edge attributes
        num_partitions: Number of data partitions

    Returns:
        Broadcast topology object with paths for each (dst, partition) pair
    '''
    pass
"""

SEED_PROGRAM = '''
"""Broadcast optimization algorithm for minimizing transfer cost across multi-cloud networks"""

import networkx as nx
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="cost")
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    return bc_topology


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

        # dict(dst) --> dict(partition) --> list(nx.edges)
        # example: {dst1: {partition1: [src->node1, node1->dst1], partition 2: [src->dst1]}}
        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        """
        Set paths for partition = partition to reach dst
        """
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        """
        Append path for partition = partition to reach dst
        """
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)

'''



# --- Evaluation ---

LOWER_COST = 1199.00
UPPER_COST = 626.24
NUM_VMS = 2

CONFIG_NAMES = [
    "intra_aws",
    "intra_azure",
    "intra_gcp",
    "inter_agz",
    "inter_gaz2",
]

_CONTEXT_CACHE: dict[str, Any] | None = None


def _resolve_resources_dir() -> Path:
    """Resolve ADRS Cloudcast resources directory."""
    env_override = os.getenv("CLOUDCAST_RESOURCES")
    if env_override:
        return Path(env_override).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent / "ADRS-Leaderboard" / "problems" / "cloudcast" / "resources"


def _load_context() -> dict[str, Any]:
    """Load and cache Cloudcast simulator resources."""
    global _CONTEXT_CACHE
    if _CONTEXT_CACHE is not None:
        return _CONTEXT_CACHE

    resources_dir = _resolve_resources_dir()
    if not resources_dir.exists():
        raise FileNotFoundError(
            "Cloudcast resources not found. Set CLOUDCAST_RESOURCES or place "
            f"resources at: {resources_dir}"
        )

    resources_str = str(resources_dir)
    if resources_str not in sys.path:
        sys.path.insert(0, resources_str)

    from simulator import BCSimulator
    from utils import make_nx_graph
    from broadcast import BroadCastTopology

    config_dir = resources_dir / "datasets" / "examples" / "config"
    cost_csv = resources_dir / "datasets" / "profiles" / "cost.csv"
    throughput_csv = resources_dir / "datasets" / "profiles" / "throughput.csv"

    config_files = {
        "intra_aws": config_dir / "intra_aws.json",
        "intra_azure": config_dir / "intra_azure.json",
        "intra_gcp": config_dir / "intra_gcp.json",
        "inter_agz": config_dir / "inter_agz.json",
        "inter_gaz2": config_dir / "inter_gaz2.json",
    }

    missing = [str(path) for path in [cost_csv, throughput_csv, *config_files.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Cloudcast files: {missing}")

    graph = make_nx_graph(
        cost_path=str(cost_csv),
        throughput_path=str(throughput_csv),
        num_vms=NUM_VMS,
    )
    configs = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in config_files.items()
    }

    _CONTEXT_CACHE = {
        "graph": graph,
        "configs": configs,
        "BCSimulator": BCSimulator,
        "BroadCastTopology": BroadCastTopology,
    }
    return _CONTEXT_CACHE


def _compute_config_score(cost: float) -> float:
    """Convert a per-config cost to a normalized [0, 1] score."""
    per_config_baseline = LOWER_COST / len(CONFIG_NAMES)
    per_config_optimal = UPPER_COST / len(CONFIG_NAMES)
    if cost >= per_config_baseline:
        return 0.0
    if cost <= per_config_optimal:
        return 1.0
    return (per_config_baseline - cost) / (per_config_baseline - per_config_optimal)


def _inject_runtime_globals(search_algorithm: Any, broad_cast_topology_cls: Any) -> list[str]:
    """Inject common globals so candidate code can run without boilerplate imports."""
    runtime_globals = search_algorithm.__globals__
    injections = {
        "BroadCastTopology": broad_cast_topology_cls,
        "nx": nx,
        "networkx": nx,
        "np": np,
        "numpy": np,
        "math": math,
        "time": time_module,
        "random": random,
        "collections": collections,
        "heapq": heapq,
        "Dict": Dict,
        "List": List,
        "Set": Set,
        "Tuple": Tuple,
        "Any": Any,
    }

    added_keys: list[str] = []
    for key, value in injections.items():
        if key not in runtime_globals:
            runtime_globals[key] = value
            added_keys.append(key)
    return added_keys


def _restore_runtime_globals(search_algorithm: Any, added_keys: list[str]) -> None:
    runtime_globals = search_algorithm.__globals__
    for key in added_keys:
        runtime_globals.pop(key, None)


def score_fn(search_algorithm: Any, _inputs: list[Any] | None = None) -> dict:
    """
    Evaluate a Cloudcast search algorithm and return a 0-100 score.

    Returns score plus per-config metrics used as behavior dimensions.
    """
    try:
        context = _load_context()
    except Exception as e:
        return {"error": f"Cloudcast setup error: {e}"}

    added_keys = _inject_runtime_globals(search_algorithm, context["BroadCastTopology"])

    per_config_costs: dict[str, float] = {}
    per_config_times: dict[str, float] = {}
    per_config_scores: dict[str, float] = {}
    total_cost = 0.0
    total_time = 0.0

    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            for config_name, config in context["configs"].items():
                try:
                    bc_topology = search_algorithm(
                        config["source_node"],
                        config["dest_nodes"],
                        context["graph"],
                        config["num_partitions"],
                    )
                    bc_topology.set_num_partitions(config["num_partitions"])

                    simulator = context["BCSimulator"](num_vms=NUM_VMS, output_dir="evals")
                    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
                        transfer_time, cost = simulator.evaluate_path(bc_topology, config)
                except Exception as e:
                    return {"error": f"Config {config_name} failed: {str(e)[:160]}"}

                if not math.isfinite(cost) or cost < 0:
                    return {"error": f"Invalid cost for {config_name}: {cost}"}
                if not math.isfinite(transfer_time) or transfer_time <= 0:
                    return {"error": f"Invalid transfer time for {config_name}: {transfer_time}"}

                per_config_costs[config_name] = float(cost)
                per_config_times[config_name] = float(transfer_time)
                per_config_scores[config_name] = _compute_config_score(float(cost))
                total_cost += float(cost)
                total_time += float(transfer_time)
    finally:
        os.chdir(original_cwd)
        _restore_runtime_globals(search_algorithm, added_keys)

    if total_cost <= 0:
        return {"error": "Invalid solution: zero total cost (likely no data transferred)"}
    if not math.isfinite(total_time) or total_time <= 0:
        return {"error": f"Invalid total transfer time: {total_time}"}

    cost_clamped = max(min(total_cost, LOWER_COST), UPPER_COST)
    normalized_cost = (LOWER_COST - cost_clamped) / (LOWER_COST - UPPER_COST)
    score = normalized_cost * 100.0

    return {
        "score": float(score),
        "total_cost": float(total_cost),
        "total_time": float(total_time),
        "successful_configs": len(per_config_costs),
        "per_config_costs": per_config_costs,
        "per_config_times": per_config_times,
        "intra_aws_score": per_config_scores.get("intra_aws", 0.0),
        "intra_azure_score": per_config_scores.get("intra_azure", 0.0),
        "intra_gcp_score": per_config_scores.get("intra_gcp", 0.0),
        "inter_agz_score": per_config_scores.get("inter_agz", 0.0),
        "inter_gaz2_score": per_config_scores.get("inter_gaz2", 0.0),
    }
