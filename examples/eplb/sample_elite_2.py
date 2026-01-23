import torch
import numpy as np


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
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
    """
    num_layers, num_logical_experts = weight.shape
    device = weight.device
    
    # Convert to numpy for processing
    weight_np = weight.cpu().numpy()
    
    # Initialize outputs
    physical_to_logical_map = torch.zeros(num_layers, num_replicas, dtype=torch.int64, device=device)
    expert_count = torch.ones(num_layers, num_logical_experts, dtype=torch.int64, device=device)
    
    # Calculate remaining replicas after base allocation
    remaining_replicas = num_replicas - num_logical_experts
    
    # Step 1: Distribute remaining replicas using load-based allocation
    for layer in range(num_layers):
        # Calculate load distribution
        layer_weight = weight_np[layer]
        total_weight = layer_weight.sum()
        
        if total_weight == 0:
            # Handle zero-weight case - distribute evenly
            additional_replicas = remaining_replicas // num_logical_experts
            extra = remaining_replicas % num_logical_experts
            expert_count[layer] += additional_replicas
            if extra > 0:
                expert_count[layer, :extra] += 1
        else:
            # Calculate weighted distribution
            # Use sqrt to moderate extreme allocations while still favoring high-weight experts
            weighted_loads = np.sqrt(layer_weight)
            normalized_loads = weighted_loads / weighted_loads.sum()
            
            # Allocate remaining replicas based on normalized loads
            exact_allocations = normalized_loads * remaining_replicas
            floor_allocations = np.floor(exact_allocations).astype(int)
            
            # Distribute remainder to experts with largest fractional parts
            fractional_parts = exact_allocations - floor_allocations
            remainder = remaining_replicas - floor_allocations.sum()
            
            if remainder > 0:
                # Sort by fractional part descending
                sorted_indices = np.argsort(fractional_parts)[::-1]
                floor_allocations[sorted_indices[:remainder]] += 1
            
            expert_count[layer] += torch.tensor(floor_allocations, dtype=torch.int64, device=device)
    
    # Step 2: Create expert-to-physical mapping
    max_replicas = expert_count.max().item()
    logical_to_physical_map = torch.full((num_layers, num_logical_experts, max_replicas), 
                                        -1, dtype=torch.int64, device=device)
    
    # Step 3: Assign physical slots using GPU-aware load balancing
    slots_per_gpu = num_replicas // num_gpus
    
    for layer in range(num_layers):
        # Track current slot allocation per GPU
        gpu_slots_used = np.zeros(num_gpus, dtype=int)
        gpu_loads = np.zeros(num_gpus, dtype=float)
        
        # Create priority queue of expert replicas by weight
        # Format: (-weight, expert_id, replica_id)
        replica_queue = []
        for expert_id in range(num_logical_experts):
            for replica_id in range(expert_count[layer, expert_id].item()):
                replica_queue.append((-weight_np[layer, expert_id], expert_id, replica_id))
        
        # Sort by weight (descending) - highest weight first
        replica_queue.sort()
        
        # Assign each replica to the GPU with minimum current load
        for neg_weight, expert_id, replica_id in replica_queue:
            # Find GPU with minimum load
            target_gpu = np.argmin(gpu_loads)
            
            # Check if GPU has available slots
            if gpu_slots_used[target_gpu] < slots_per_gpu:
                # Assign to this GPU
                physical_slot = target_gpu * slots_per_gpu + gpu_slots_used[target_gpu]
                
                physical_to_logical_map[layer, physical_slot] = expert_id
                logical_to_physical_map[layer, expert_id, replica_id] = physical_slot
                
                # Update GPU state
                gpu_loads[target_gpu] += -neg_weight
                gpu_slots_used[target_gpu] += 1
            else:
                # Find next available GPU (shouldn't happen with proper load balancing)
                for gpu_id in range(num_gpus):
                    if gpu_slots_used[gpu_id] < slots_per_gpu:
                        physical_slot = gpu_id * slots_per_gpu + gpu_slots_used[gpu_id]
                        physical_to_logical_map[layer, physical_slot] = expert_id
                        logical_to_physical_map[layer, expert_id, replica_id] = physical_slot
                        gpu_loads[gpu_id] += -neg_weight
                        gpu_slots_used[gpu_id] += 1
                        break
    
    # Verify constraints
    assert expert_count.sum(dim=1).eq(num_replicas).all(), "Total replicas must equal num_replicas"
    assert (expert_count > 0).all(), "All experts must have at least one replica"
    assert (physical_to_logical_map >= 0).all() and (physical_to_logical_map < num_logical_experts).all(), \
           "Physical to logical mapping contains invalid indices"
    assert (logical_to_physical_map >= -1).all(), "Logical to physical mapping contains invalid indices"
    assert ((logical_to_physical_map >= 0) | (logical_to_physical_map == -1)).all(), \
           "Logical to physical mapping should only contain valid slots or -1"
    
    return physical_to_logical_map, logical_to_physical_map, expert_count