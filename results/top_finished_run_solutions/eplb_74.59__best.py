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
    Optimized for load balance using Greedy Apportionment and Snake Mapping.

    Parameters:
        weight: [layers, 64] load statistics
        num_replicas: 288 (physical experts)
        num_groups: 8
        num_nodes: 4
        num_gpus: 32

    Returns:
        physical_to_logical_map: [layers, 288] (values 0-63)
        logical_to_physical_map: [layers, 64, X] (physical indices or -1)
        expert_count: [layers, 64] (number of replicas per expert)
    '''
    num_layers, num_logical = weight.shape
    device = weight.device
    slots_per_gpu = num_replicas // num_gpus # 9
    
    # --- 1. Expert Apportionment (Greedy) ---
    # Every expert must have at least 1 replica
    expert_count = torch.ones((num_layers, num_logical), dtype=torch.int64, device=device)
    
    # Remaining 224 slots per layer
    remaining = num_replicas - num_logical
    
    # Work with float64 for precision
    w_float = weight.to(torch.float64) + 1e-12
    current_counts = expert_count.clone().to(torch.float64)
    
    # Assign remaining slots to experts with highest load-per-replica
    # Vectorized across layers
    for _ in range(remaining):
        priority = w_float / current_counts
        best_expert = torch.argmax(priority, dim=1)
        row_indices = torch.arange(num_layers, device=device)
        expert_count[row_indices, best_expert] += 1
        current_counts[row_indices, best_expert] += 1.0

    # --- 2. Map Generation ---
    # Logical IDs and their replica ranks (0, 1, 2...) for every physical slot
    # sorted by load to allow balanced assignment
    rep_load_per_expert = w_float / current_counts
    
    # Expand logical experts into a flat list of items per layer
    # expert_offsets: [layers, 65]
    expert_offsets = torch.zeros((num_layers, num_logical + 1), dtype=torch.int64, device=device)
    expert_offsets[:, 1:] = torch.cumsum(expert_count, dim=1)
    
    # seq: [layers, 288] -> maps flat index to logical expert id
    seq = torch.arange(num_replicas, device=device).expand(num_layers, -1)
    logical_ids = torch.searchsorted(expert_offsets, seq, right=True) - 1
    logical_ids = torch.clamp(logical_ids, 0, num_logical - 1)
    
    # Calculate the instance rank (which replica it is for that expert)
    # rank: [layers, 288]
    ranks = seq - torch.gather(expert_offsets, 1, logical_ids)
    
    # Get load for each individual replica
    replica_loads = torch.gather(rep_load_per_expert, 1, logical_ids)
    
    # Sort replicas by load descending
    sort_idx = torch.argsort(replica_loads, dim=1, descending=True)
    sorted_logical = torch.gather(logical_ids, 1, sort_idx)
    sorted_ranks = torch.gather(ranks, 1, sort_idx)

    # --- 3. Snake Mapping for Load Balancing ---
    # We map sorted replicas to GPUs using a zig-zag pattern
    # GPU 0..31, then 31..0, then 0..31...
    gpu_indices = torch.arange(num_gpus, device=device)
    placement_list = []
    for s in range(slots_per_gpu):
        if s % 2 == 0:
            order = gpu_indices
        else:
            order = gpu_indices.flip(0)
        # Physical index = gpu_id * 9 + slot_in_gpu
        placement_list.append(order * slots_per_gpu + s)
    
    # placement_order: [288]
    placement_order = torch.cat(placement_list)
    
    physical_to_logical_map = torch.zeros((num_layers, num_replicas), dtype=torch.int64, device=device)
    physical_rank_map = torch.zeros((num_layers, num_replicas), dtype=torch.int64, device=device)
    
    # Scatter the sorted items into the physical slots defined by snake order
    dest = placement_order.expand(num_layers, -1)
    physical_to_logical_map.scatter_(1, dest, sorted_logical)
    physical_rank_map.scatter_(1, dest, sorted_ranks)

    # --- 4. Inverse Map (Logical to Physical) ---
    max_reps = int(expert_count.max().item())
    logical_to_physical_map = torch.full((num_layers, num_logical, max_reps), -1, dtype=torch.int64, device=device)
    
    # Prepare indices for scatter
    layer_idx = torch.arange(num_layers, device=device).unsqueeze(1).expand(-1, num_replicas)
    phys_idx = torch.arange(num_replicas, device=device).expand(num_layers, -1)
    
    # Flattened index for 3D tensor: [layer, expert, rank]
    flat_dest_indices = (
        layer_idx * (num_logical * max_reps) +
        physical_to_logical_map * max_reps +
        physical_rank_map
    ).reshape(-1)
    
    logical_to_physical_map.view(-1).scatter_(0, flat_dest_indices, phys_idx.reshape(-1))

    return physical_to_logical_map, logical_to_physical_map, expert_count