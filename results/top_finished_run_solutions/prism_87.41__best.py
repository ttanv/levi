import collections
import time
import random
from dataclasses import dataclass

GPU_MEM_SIZE = 80  # GB

@dataclass
class Model:
    model_name: str
    model_size: int   # GB
    req_rate: int     # requests per second
    slo: int          # service level objective (latency target)
    cur_gpu_id: int   # current GPU assignment (can be ignored)

def compute_model_placement(gpu_num: int, models: list[Model]) -> dict[int, list[Model]]:
    """
    Optimizes model placement to minimize max KVPR.
    Strategy:
    1. Initial Greedy Packing using a Pressure-Aware First-Fit.
    2. Simulated Annealing / Local Search to refine the bottleneck (max pressure).
    """

    def calculate_kvpr(gpu_models):
        if not gpu_models:
            return 0.0
        total_weight = sum(m.req_rate / m.slo for m in gpu_models)
        used_mem = sum(m.model_size for m in gpu_models)
        remaining_mem = GPU_MEM_SIZE - used_mem
        # Respect hard constraint; return inf if exceeded
        if remaining_mem <= 0:
            return float('inf')
        return total_weight / remaining_mem

    # Initial Greedy Placement: Sort models by weight/size density
    # This helps in packing efficiently against the KVPR formula
    sorted_models = sorted(models, key=lambda x: (x.req_rate / x.slo) / x.model_size, reverse=True)
    
    placement = {i: [] for i in range(gpu_num)}
    mem_used = [0] * gpu_num
    
    # Simple First-Fit Decreasing for memory
    for m in sorted_models:
        placed = False
        # Try to place in the GPU with the most remaining memory to keep KVPR low
        best_gpu = -1
        max_rem = -1
        for i in range(gpu_num):
            if mem_used[i] + m.model_size <= GPU_MEM_SIZE:
                rem = GPU_MEM_SIZE - (mem_used[i] + m.model_size)
                if rem > max_rem:
                    max_rem = rem
                    best_gpu = i
        
        if best_gpu != -1:
            placement[best_gpu].append(m)
            mem_used[best_gpu] += m.model_size
        else:
            # Fallback (should not happen given constraints): place in first valid
            for i in range(gpu_num):
                if mem_used[i] + m.model_size <= GPU_MEM_SIZE:
                    placement[i].append(m)
                    mem_used[i] += m.model_size
                    break

    # Local Search Refinement
    start_time = time.time()
    # Time limit is 10s per test, but we use a safe margin for 50 tests
    # Actually, the problem says 10s per test, so 1-2s is very safe.
    timeout = 1.5 

    best_max_kvpr = max(calculate_kvpr(placement[i]) for i in range(gpu_num))
    best_placement = {i: list(placement[i]) for i in range(gpu_num)}

    while time.time() - start_time < timeout:
        # Calculate current pressures
        pressures = {i: calculate_kvpr(placement[i]) for i in range(gpu_num)}
        max_gpu = max(pressures, key=pressures.get)
        min_gpu = min(pressures, key=pressures.get)
        
        current_max = pressures[max_gpu]
        if current_max < best_max_kvpr:
            best_max_kvpr = current_max
            best_placement = {i: list(placement[i]) for i in range(gpu_num)}

        # Try to move a model from the most pressured GPU to another
        moved = False
        if placement[max_gpu]:
            target_gpus = sorted(range(gpu_num), key=lambda x: pressures[x])
            for target_gpu in target_gpus:
                if target_gpu == max_gpu: continue
                
                # Try moving one model
                for idx, m in enumerate(placement[max_gpu]):
                    if sum(x.model_size for x in placement[target_gpu]) + m.model_size <= GPU_MEM_SIZE:
                        # Trial move
                        old_max_list = placement[max_gpu]
                        old_target_list = placement[target_gpu]
                        
                        new_max_list = old_max_list[:idx] + old_max_list[idx+1:]
                        new_target_list = old_target_list + [m]
                        
                        p1 = calculate_kvpr(new_max_list)
                        p2 = calculate_kvpr(new_target_list)
                        
                        if max(p1, p2) < current_max:
                            placement[max_gpu] = new_max_list
                            placement[target_gpu] = new_target_list
                            moved = True
                            break
                if moved: break

        # If move didn't work, try a swap
        if not moved:
            other_gpus = list(range(gpu_num))
            random.shuffle(other_gpus)
            for other_gpu in other_gpus:
                if other_gpu == max_gpu: continue
                
                for i, m_max in enumerate(placement[max_gpu]):
                    for j, m_other in enumerate(placement[other_gpu]):
                        # Check memory constraints for swap
                        new_mem_max = sum(x.model_size for x in placement[max_gpu]) - m_max.model_size + m_other.model_size
                        new_mem_other = sum(x.model_size for x in placement[other_gpu]) - m_other.model_size + m_max.model_size
                        
                        if new_mem_max <= GPU_MEM_SIZE and new_mem_other <= GPU_MEM_SIZE:
                            # Trial swap
                            temp_max = [x for k, x in enumerate(placement[max_gpu]) if k != i] + [m_other]
                            temp_other = [x for k, x in enumerate(placement[other_gpu]) if k != j] + [m_max]
                            
                            p1 = calculate_kvpr(temp_max)
                            p2 = calculate_kvpr(temp_other)
                            
                            if max(p1, p2) < current_max:
                                placement[max_gpu] = temp_max
                                placement[other_gpu] = temp_other
                                moved = True
                                break
                    if moved: break
                if moved: break
        
        # If still no improvement, perform a random perturbation to escape local minima
        if not moved:
            g1, g2 = random.sample(range(gpu_num), 2)
            if placement[g1]:
                idx = random.randrange(len(placement[g1]))
                m = placement[g1][idx]
                if sum(x.model_size for x in placement[g2]) + m.model_size <= GPU_MEM_SIZE:
                    placement[g2].append(placement[g1].pop(idx))

    return best_placement