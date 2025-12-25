"""
Prompts for EPLB (Expert Parallelism Load Balancer) evolution.
"""

PROBLEM_DESCRIPTION = """
# Expert Parallelism Load Balancer (EPLB)

## Goal
Map 256 logical experts to 288 physical GPU slots for optimal load balancing.
Score = balancedness (0-90 pts) + speed (0-10 pts). Target: 88+ points.

## Fixed Constants
- num_layers = 58 (from weight.shape[0])
- num_logical_experts = 256 (from weight.shape[1])
- num_replicas = 288 (physical slots)
- num_gpus = 32, num_nodes = 4, num_groups = 8

## EXACT Output Requirements
Your function MUST return exactly these shapes:
1. phy2log: torch.int64, shape [58, 288] - values in range [0, 255]
2. log2phy: torch.int64, shape [58, 256, K] where K = max replicas per expert
3. logcnt: torch.int64, shape [58, 256] - sum per layer MUST equal 288

## Scoring (target 88+)
- balancedness = mean(gpu_avg_load / gpu_max_load) * 90
- speed_bonus = min(0.002 / exec_time, 2) * 5
- slow_penalty = max(0, exec_time - 0.01) * 20 (if > 10ms)

## MANDATORY: Use This Template Structure
```python
import torch

def rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus):
    num_layers, num_logical_experts = weight.shape  # [58, 256]
    weight = weight.float()

    # Step 1: Compute replica counts (must sum to 288 per layer)
    logcnt = torch.ones(num_layers, num_logical_experts, dtype=torch.int64)
    # ... add logic to distribute remaining 288-256=32 replicas ...

    # Step 2: Build phy2log [58, 288] from logcnt
    phy2log = torch.zeros(num_layers, num_replicas, dtype=torch.int64)
    for layer in range(num_layers):
        idx = 0
        for log_id in range(num_logical_experts):
            cnt = logcnt[layer, log_id].item()
            phy2log[layer, idx:idx+cnt] = log_id
            idx += cnt

    # Step 3: Build log2phy [58, 256, max_cnt] as inverse mapping
    max_cnt = logcnt.max().item()
    log2phy = torch.full((num_layers, num_logical_experts, max_cnt), -1, dtype=torch.int64)
    for layer in range(num_layers):
        idx = 0
        for log_id in range(num_logical_experts):
            cnt = logcnt[layer, log_id].item()
            log2phy[layer, log_id, :cnt] = torch.arange(idx, idx + cnt)
            idx += cnt

    return phy2log, log2phy, logcnt
```

## Key Optimization Ideas (pick one approach)
1. **Greedy**: Give extra replicas to highest-load experts
2. **Proportional**: Distribute replicas proportional to load
3. **GPU-aware**: Balance load across 32 GPUs (9 slots each)
4. **Vectorized**: Replace Python loops with torch ops for speed

## ZERO TOLERANCE ERRORS - Your code will FAIL if:
- phy2log.shape != [58, 288]
- logcnt.sum(dim=1) != 288 for any layer
- Any index >= 288 or >= 256 (out of bounds)
- Using CUDA (.cuda(), torch.cuda.*)
- Tensor dimension mismatches in scatter/gather

Output ONLY the Python code block. No explanations.
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

# Inspiration seeds varying in complexity and approach
SEED_INSPIRATIONS = [
    # === SIMPLE SEEDS ===
    # 1. Minimal: uniform distribution (very simple baseline)
    '''import torch

def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple uniform distribution: each logical expert gets equal replicas."""
    num_layers, num_logical_experts = weight.shape

    # Each logical expert gets at least 1 replica
    base_count = num_replicas // num_logical_experts
    extra = num_replicas % num_logical_experts

    logcnt = torch.full((num_layers, num_logical_experts), base_count, dtype=torch.int64)
    logcnt[:, :extra] += 1

    # Build physical to logical mapping
    phy2log = torch.zeros(num_layers, num_replicas, dtype=torch.int64)
    idx = 0
    for log_id in range(num_logical_experts):
        cnt = logcnt[0, log_id].item()
        phy2log[:, idx:idx+cnt] = log_id
        idx += cnt

    # Build logical to physical mapping
    maxlogcnt = logcnt.max().item()
    log2phy = torch.full((num_layers, num_logical_experts, maxlogcnt), -1, dtype=torch.int64)
    for layer in range(num_layers):
        idx = 0
        for log_id in range(num_logical_experts):
            cnt = logcnt[layer, log_id].item()
            log2phy[layer, log_id, :cnt] = torch.arange(idx, idx + cnt)
            idx += cnt

    return phy2log, log2phy, logcnt
''',

    # 2. Proportional replication based on load
    '''import torch

def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Proportional: more replicas for higher-load experts."""
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()

    # Compute replica count proportional to load
    total_weight = weight.sum(dim=-1, keepdim=True)
    proportions = weight / total_weight.clamp(min=1e-6)

    # Each expert gets at least 1 replica
    logcnt = torch.ones(num_layers, num_logical_experts, dtype=torch.int64)
    remaining = num_replicas - num_logical_experts

    # Distribute remaining replicas proportionally
    extra_replicas = (proportions * remaining).floor().long()
    logcnt += extra_replicas

    # Distribute any remaining
    current_total = logcnt.sum(dim=-1)
    for layer in range(num_layers):
        diff = num_replicas - current_total[layer].item()
        if diff > 0:
            # Give to highest load experts
            top_indices = weight[layer].argsort(descending=True)[:diff]
            logcnt[layer, top_indices] += 1

    # Build mappings
    phy2log = torch.zeros(num_layers, num_replicas, dtype=torch.int64)
    maxlogcnt = logcnt.max().item()
    log2phy = torch.full((num_layers, num_logical_experts, maxlogcnt), -1, dtype=torch.int64)

    for layer in range(num_layers):
        idx = 0
        for log_id in range(num_logical_experts):
            cnt = logcnt[layer, log_id].item()
            phy2log[layer, idx:idx+cnt] = log_id
            log2phy[layer, log_id, :cnt] = torch.arange(idx, idx + cnt)
            idx += cnt

    return phy2log, log2phy, logcnt
''',

    # 3. Greedy max-load reduction
    '''import torch

def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy: iteratively add replicas to highest-load expert."""
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()

    logcnt = torch.ones(num_layers, num_logical_experts, dtype=torch.int64)

    # Greedily assign remaining replicas
    for _ in range(num_replicas - num_logical_experts):
        # Find expert with highest load per replica
        load_per_replica = weight / logcnt.float()
        max_indices = load_per_replica.argmax(dim=-1)
        for layer in range(num_layers):
            logcnt[layer, max_indices[layer]] += 1

    # Build mappings
    phy2log = torch.zeros(num_layers, num_replicas, dtype=torch.int64)
    maxlogcnt = logcnt.max().item()
    log2phy = torch.full((num_layers, num_logical_experts, maxlogcnt), -1, dtype=torch.int64)

    for layer in range(num_layers):
        idx = 0
        for log_id in range(num_logical_experts):
            cnt = logcnt[layer, log_id].item()
            phy2log[layer, idx:idx+cnt] = log_id
            log2phy[layer, log_id, :cnt] = torch.arange(idx, idx + cnt)
            idx += cnt

    return phy2log, log2phy, logcnt
''',

    # 4. GPU-aware round-robin
    '''import torch

def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU-aware: distribute experts evenly across GPUs."""
    num_layers, num_logical_experts = weight.shape
    experts_per_gpu = num_replicas // num_gpus

    # Start with uniform distribution
    logcnt = torch.ones(num_layers, num_logical_experts, dtype=torch.int64)

    # Build phy2log with GPU awareness
    phy2log = torch.zeros(num_layers, num_replicas, dtype=torch.int64)

    for layer in range(num_layers):
        # Sort experts by load (descending)
        sorted_indices = weight[layer].argsort(descending=True)

        # Assign to physical slots in GPU-interleaved order
        for phy_id in range(num_replicas):
            gpu_id = phy_id // experts_per_gpu
            slot_in_gpu = phy_id % experts_per_gpu
            # Pick expert based on slot, cycling through sorted experts
            log_id = sorted_indices[phy_id % num_logical_experts].item()
            phy2log[layer, phy_id] = log_id
            logcnt[layer, log_id] += 1

    # Adjust logcnt (we overcounted by 1 initially)
    logcnt -= 1

    # Build log2phy
    maxlogcnt = logcnt.max().item()
    log2phy = torch.full((num_layers, num_logical_experts, maxlogcnt), -1, dtype=torch.int64)

    for layer in range(num_layers):
        counts = torch.zeros(num_logical_experts, dtype=torch.int64)
        for phy_id in range(num_replicas):
            log_id = phy2log[layer, phy_id].item()
            rank = counts[log_id].item()
            if rank < maxlogcnt:
                log2phy[layer, log_id, rank] = phy_id
            counts[log_id] += 1

    return phy2log, log2phy, logcnt
''',

    # 5. Vectorized greedy (faster)
    '''import torch

def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()

    logcnt = torch.ones(num_layers, num_logical_experts, dtype=torch.int64)
    remaining = num_replicas - num_logical_experts

    for _ in range(remaining):
        load_per_replica = weight / logcnt.float()
        max_idx = load_per_replica.argmax(dim=-1)
        logcnt.scatter_add_(1, max_idx.unsqueeze(1),
                           torch.ones(num_layers, 1, dtype=torch.int64))

    phy2log_list = []
    for layer in range(num_layers):
        expert_ids = torch.arange(num_logical_experts)
        phy2log_layer = expert_ids.repeat_interleave(logcnt[layer])
        phy2log_list.append(phy2log_layer)
    phy2log = torch.stack(phy2log_list)

    maxlogcnt = logcnt.max().item()
    log2phy = torch.full((num_layers, num_logical_experts, maxlogcnt), -1, dtype=torch.int64)

    for layer in range(num_layers):
        idx = 0
        for log_id in range(num_logical_experts):
            cnt = logcnt[layer, log_id].item()
            log2phy[layer, log_id, :cnt] = torch.arange(idx, idx + cnt)
            idx += cnt

    return phy2log, log2phy, logcnt
''',
]
