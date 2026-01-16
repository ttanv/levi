"""
EPLB (Expert Parallelism Load Balancer) Problem Definition.

Contains problem description, prompts, scoring function, and test inputs.
"""

import os
import json
import time
from pathlib import Path

import torch

# --- EPLB Constants ---
NUM_REPLICAS = 288
NUM_GROUPS = 8
NUM_GPUS = 32
NUM_NODES = 4
REBALANCE_INTERVAL = 100

# --- Prompts ---
PROBLEM_DESCRIPTION = """
# EPLB Problem

Optimize expert placement for MoE models. Map logical experts to physical GPU slots for load balancing.

## Input
- `weight`: [layers, num_logical_experts] - load per logical expert (typically 64 logical experts)
- `num_replicas`: 288 total physical slots
- `num_gpus`: 32 GPUs (each GPU has 288/32 = 9 slots)
- `num_groups`: 8, `num_nodes`: 4

## Output (all torch.int64 tensors)
- `physical_to_logical_map`: [layers, 288] - which logical expert in each physical slot (values 0 to num_logical_experts-1)
- `logical_to_physical_map`: [layers, num_logical_experts, max_replicas] - physical slots for each logical expert, padded with -1
- `expert_count`: [layers, num_logical_experts] - replication count per logical expert (sum must equal 288)

## Constraints
- Every physical slot must be assigned (no -1 in physical_to_logical_map)
- expert_count[layer].sum() == 288 for all layers
- Heavily-loaded experts should have more replicas

## Scoring
- 90% balancedness: avg_gpu_load / max_gpu_load (GPU load = sum of weights of experts on that GPU)
- 10% speed: faster is better, penalty if >10ms

Output ONLY Python code in a ```python block.
"""

FUNCTION_SIGNATURE = """
import torch

def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Rearrange and replicate logical experts across physical GPU slots.

    Parameters:
        weight: [layers, num_logical_experts], load statistics
        num_replicas: 288 (physical experts)
        num_groups: 8
        num_nodes: 4
        num_gpus: 32

    Returns:
        physical_to_logical_map: [layers, num_replicas]
        logical_to_physical_map: [layers, num_logical_experts, X]
        expert_count: [layers, num_logical_experts]
    '''
    pass
"""

SEED_PROGRAM = '''"""Expert parallelism load balancer for MoE models in distributed inference."""

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight,
                                 fill_value=-1,
                                 dtype=torch.int64,
                                 device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (i
                 for i in range(num_packs) if pack_items[i] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64,
                           device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """Hierarchical load balancing: nodes -> GPUs -> experts."""
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64,
                         device=perm.device).expand(perm.shape),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=group_pack_index.device,
    ).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=log2phy.device).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt
'''

SEED_INSPIRATIONS = []

# --- Workload Loading ---
_WORKLOADS = None


def _find_workload_path() -> str:
    """Find the expert-load.json file."""
    # Try Docker path first
    docker_path = "/datasets/eplb/expert-load.json"
    if os.path.exists(docker_path):
        return docker_path

    # Try relative to this file
    this_dir = Path(__file__).parent
    possible_paths = [
        this_dir.parent.parent.parent / "ADRS-Leaderboard" / "datasets" / "eplb" / "expert-load.json",
        this_dir.parent.parent.parent.parent / "ADRS-Leaderboard" / "datasets" / "eplb" / "expert-load.json",
        Path.home() / "ADRS-Leaderboard" / "datasets" / "eplb" / "expert-load.json",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"expert-load.json not found. Tried: {docker_path}, {[str(p) for p in possible_paths]}"
    )


def _load_workloads():
    """Load workloads from JSON file (cached)."""
    global _WORKLOADS
    if _WORKLOADS is None:
        workload_path = _find_workload_path()
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


def get_inputs():
    """Return the workloads as inputs for the scoring function."""
    return _load_workloads()


# --- Simulation ---
def simulate_inference(log2phy, logcnt, workload):
    """Simulate MoE inference and return balancedness scores."""
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


# --- Score Function ---
def score_fn(rebalance_experts, inputs):
    """
    Evaluate EPLB algorithm.

    Args:
        rebalance_experts: The function to evaluate
        inputs: List of workload tensors

    Returns:
        dict with 'score' or 'error' key
    """
    try:
        workloads = inputs

        balancedness_scores_gpu = []
        balancedness_scores_expert = []
        times_algorithm = []

        for i in range(len(workloads) - 1):
            start_time = time.perf_counter()
            phy2log, log2phy, logcnt = rebalance_experts(
                workloads[i],
                NUM_REPLICAS,
                NUM_GROUPS,
                NUM_NODES,
                NUM_GPUS,
            )
            end_time = time.perf_counter()

            # Validate outputs
            if phy2log.shape[1] != NUM_REPLICAS:
                return {"error": f"phy2log shape wrong: {phy2log.shape}"}

            if not torch.all(logcnt.sum(dim=1) == NUM_REPLICAS):
                sums = logcnt.sum(dim=1)
                return {"error": f"logcnt sums != {NUM_REPLICAS}: {sums[:5].tolist()}..."}

            # Check for negative replica counts (could bypass sum check)
            if (logcnt < 0).any():
                return {"error": "logcnt contains negative values"}

            # Check for unhandled load: experts with load but 0 replicas
            next_workload = workloads[i + 1]
            has_load = next_workload > 0
            has_no_replicas = logcnt == 0
            unhandled = has_load & has_no_replicas
            if unhandled.any():
                unhandled_count = unhandled.sum().item()
                return {"error": f"Unhandled load: {unhandled_count} experts have load but 0 replicas"}

            balancedness_gpu, balancedness_expert = simulate_inference(
                log2phy, logcnt, workloads[i + 1]
            )

            balancedness_scores_gpu.append(balancedness_gpu)
            balancedness_scores_expert.append(balancedness_expert)
            times_algorithm.append(end_time - start_time)

        avg_balancedness_gpu = sum(balancedness_scores_gpu) / len(balancedness_scores_gpu)
        avg_balancedness_expert = sum(balancedness_scores_expert) / len(balancedness_scores_expert)
        avg_time = sum(times_algorithm) / len(times_algorithm)

        # Scoring formula from evaluator
        balancedness_score = avg_balancedness_gpu * 90
        speed_raw = 0.002 / avg_time if avg_time > 0 else 2.0
        speed_capped = min(speed_raw, 2.0)
        speed_score = speed_capped * 5

        if avg_time > 0.01:  # > 10ms
            slow_penalty = min(avg_time * 20, 20)
        else:
            slow_penalty = 0

        score = balancedness_score + speed_score - slow_penalty

        # Per-workload behavioral dimensions (W8, W9 least correlated; W1-W7 cluster)
        workload_0 = balancedness_scores_gpu[0] if len(balancedness_scores_gpu) > 0 else 0.0
        workload_8 = balancedness_scores_gpu[8] if len(balancedness_scores_gpu) > 8 else 0.0
        workload_9 = balancedness_scores_gpu[9] if len(balancedness_scores_gpu) > 9 else 0.0
        main_cluster = balancedness_scores_gpu[1:8] if len(balancedness_scores_gpu) > 7 else []
        workload_main = sum(main_cluster) / len(main_cluster) if main_cluster else 0.0

        return {
            "score": score,
            "balancedness_gpu": avg_balancedness_gpu,
            "balancedness_expert": avg_balancedness_expert,
            "avg_time": avg_time,
            "execution_time": avg_time,
            "balancedness_score": balancedness_score,
            "speed_score": speed_score,
            "slow_penalty": slow_penalty,
            "workload_0": workload_0,
            "workload_8": workload_8,
            "workload_9": workload_9,
            "workload_main": workload_main,
        }
    except Exception as e:
        return {"error": str(e)}


# --- Test Inputs (lazy loaded) ---
INPUTS = None


def get_lazy_inputs():
    """Get inputs, loading them lazily on first access."""
    global INPUTS
    if INPUTS is None:
        INPUTS = get_inputs()
    return INPUTS
