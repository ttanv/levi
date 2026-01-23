"""Expert parallelism load balancer for MoE models in distributed inference."""

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

    # Use greedy packing with priority queue for efficiency
    indices = weight.float().sort(-1, descending=True).indices
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device=weight.device)
    rank_in_pack = torch.full_like(pack_index, fill_value=-1, dtype=torch.int64)

    # Pre-allocate pack state
    pack_weights = torch.zeros(num_layers, num_packs, dtype=weight.dtype, device=weight.device)
    pack_items = torch.zeros(num_layers, num_packs, dtype=torch.int64, device=weight.device)

    for group in indices.unbind(-1):
        # Find the pack with minimum weight among those not yet full
        valid_packs = pack_items < groups_per_pack
        pack_scores = torch.where(valid_packs, pack_weights, torch.inf)
        pack = pack_scores.argmin(dim=-1)

        # Update pack state
        pack_index[torch.arange(num_layers), group] = pack
        rank_in_pack[torch.arange(num_layers), group] = pack_items[torch.arange(num_layers), pack]
        pack_weights[torch.arange(num_layers), pack] += weight[torch.arange(num_layers), group]
        pack_items[torch.arange(num_layers), pack] += 1

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
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)

    # Use priority queue for efficient greedy replication
    for i in range(num_log, num_phy):
        # Compute current load per expert (weight / count)
        load_per_expert = weight / logcnt
        # Find expert with maximum load
        redundant_indices = load_per_expert.max(dim=-1).indices
        # Assign new replica
        phy2log[arangen, i] = redundant_indices
        rank[arangen, i] = logcnt[arangen, redundant_indices]
        # Update count
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
    # Map logical experts to node-local experts
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: replicate experts within each node
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs
    # Compute load per physical expert
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    # Pack to GPUs
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    # Map packed indices to physical slots
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    # Map back to logical experts
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    # Offset by node group
    offset = torch.arange(0, num_logical_experts, num_logical_experts // num_nodes,
                          dtype=torch.int64, device=group_pack_index.device).view(1, -1, 1)
    pphy2mlog = pphy2mlog.view(num_layers, num_nodes, -1) + offset
    pphy2mlog = pphy2mlog.flatten(-2)
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
    # Use float32 for computations to avoid overflow
    weight = weight.float()
    device = weight.device

    # Use hierarchical balancing if possible
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # Fallback to simple replication if hierarchical not applicable
        phy2log, phyrank, logcnt = replicate_experts(
            weight, num_replicas)

    # Compute max number of replicas per logical expert
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1

    # Build logical_to_physical_map with padding
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=device,
    )
    # Scatter physical slot indices to logical experts
    indices = phy2log * maxlogcnt + phyrank
    log2phy.view(num_layers, -1).scatter_(
        -1,
        indices,
        torch.arange(num_replicas, dtype=torch.int64, device=device).expand(num_layers, -1),
    )

    # Ensure expert_count matches sum of replicas
    assert (logcnt.sum(-1) == num_replicas).all(), "Total replicas must equal num_replicas"

    return phy2log, log2phy, logcnt
