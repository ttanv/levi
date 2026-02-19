import networkx as nx
from typing import List, Dict, Tuple
from collections import defaultdict
import heapq

def search_algorithm(src: str, dsts: List[str], G: nx.DiGraph, num_partitions: int) -> BroadCastTopology:
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    
    # Precompute all-pairs shortest paths using Dijkstra from important nodes
    all_nodes = list(G.nodes())
    
    # Compute shortest paths from source to all nodes
    src_dist = {}
    src_prev = {}
    src_dist[src] = 0
    pq = [(0, src)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > src_dist.get(u, float('inf')):
            continue
        for v in G[u]:
            edge_cost = G[u][v]['cost']
            new_dist = d + edge_cost
            if new_dist < src_dist.get(v, float('inf')):
                src_dist[v] = new_dist
                src_prev[v] = u
                heapq.heappush(pq, (new_dist, v))
    
    # Reconstruct source paths
    src_paths = {}
    for dst in dsts:
        if dst in src_dist:
            path = []
            cur = dst
            while cur != src:
                path.append(cur)
                cur = src_prev[cur]
            path.append(src)
            src_paths[dst] = path[::-1]
    
    # Group destinations by cloud and find potential hubs
    cloud_dests = defaultdict(list)
    cloud_nodes = defaultdict(list)
    
    for node in all_nodes:
        if node.startswith("aws"):
            cloud_nodes["aws"].append(node)
        elif node.startswith("azure"):
            cloud_nodes["azure"].append(node)
        elif node.startswith("gcp"):
            cloud_nodes["gcp"].append(node)
    
    for dst in dsts:
        if dst.startswith("aws"):
            cloud_dests["aws"].append(dst)
        elif dst.startswith("azure"):
            cloud_dests["azure"].append(dst)
        elif dst.startswith("gcp"):
            cloud_dests["gcp"].append(dst)
    
    # Find best hub for each cloud considering partition replication
    cloud_hubs = {}
    hub_to_dest_paths = {}
    
    for cloud in ["aws", "azure", "gcp"]:
        dests = cloud_dests[cloud]
        if len(dests) <= 1:
            continue
            
        nodes = cloud_nodes[cloud]
        best_hub = None
        best_cost = float('inf')
        best_paths = {}
        
        for hub in nodes:
            if hub not in src_dist:
                continue
            
            # Compute shortest paths from hub to destinations
            hub_dist = {}
            hub_prev = {}
            hub_dist[hub] = 0
            pq_hub = [(0, hub)]
            
            while pq_hub:
                d, u = heapq.heappop(pq_hub)
                if d > hub_dist.get(u, float('inf')):
                    continue
                for v in G[u]:
                    edge_cost = G[u][v]['cost']
                    new_dist = d + edge_cost
                    if new_dist < hub_dist.get(v, float('inf')):
                        hub_dist[v] = new_dist
                        hub_prev[v] = u
                        heapq.heappush(pq_hub, (new_dist, v))
            
            # Calculate total cost for this hub
            total_cost = src_dist[hub] * num_partitions  # Source to hub for all partitions
            
            valid = True
            paths = {}
            for dst in dests:
                if dst == hub:
                    continue
                if dst not in hub_dist:
                    valid = False
                    break
                total_cost += hub_dist[dst] * num_partitions
                
                # Reconstruct path
                path = []
                cur = dst
                while cur != hub:
                    path.append(cur)
                    cur = hub_prev[cur]
                path.append(hub)
                paths[dst] = path[::-1]
            
            if valid and total_cost < best_cost:
                best_cost = total_cost
                best_hub = hub
                best_paths = paths
        
        if best_hub:
            cloud_hubs[cloud] = best_hub
            hub_to_dest_paths[best_hub] = best_paths
    
    # Calculate costs for different strategies
    direct_cost = sum(src_dist.get(d, float('inf')) * num_partitions for d in dsts)
    
    hub_cost = 0
    hub_assignment = defaultdict(list)
    
    for cloud in ["aws", "azure", "gcp"]:
        dests = cloud_dests[cloud]
        if cloud in cloud_hubs and len(dests) > 1:
            hub = cloud_hubs[cloud]
            hub_cost += src_dist[hub] * num_partitions
            hub_assignment[hub].extend(dests)
            for dst in dests:
                if dst != hub:
                    try:
                        path_cost = nx.dijkstra_path_length(G, hub, dst, weight='cost')
                        hub_cost += path_cost * num_partitions
                    except:
                        hub_cost += src_dist.get(dst, float('inf')) * num_partitions
        else:
            for dst in dests:
                hub_cost += src_dist.get(dst, float('inf')) * num_partitions
                hub_assignment[dst].append(dst)  # Self-hub
    
    # Also consider mixed strategy: use hubs only when beneficial
    use_hubs = hub_cost < direct_cost
    
    # Build paths
    for partition in range(num_partitions):
        if use_hubs:
            processed_hubs = set()
            for hub, assigned_dests in hub_assignment.items():
                if hub in processed_hubs:
                    continue
                    
                # Add source-to-hub path for first destination
                if hub in src_paths:
                    hub_path = src_paths[hub]
                    first_dst = assigned_dests[0]
                    for i in range(len(hub_path)-1):
                        u, v = hub_path[i], hub_path[i+1]
                        bc_topology.append_dst_partition_path(first_dst, partition, [u, v, G[u][v]])
                
                # Add hub-to-destination paths
                for dst in assigned_dests:
                    if dst == hub:
                        continue
                    
                    if hub in hub_to_dest_paths and dst in hub_to_dest_paths[hub]:
                        path = hub_to_dest_paths[hub][dst]
                        for i in range(len(path)-1):
                            u, v = path[i], path[i+1]
                            bc_topology.append_dst_partition_path(dst, partition, [u, v, G[u][v]])
                    else:
                        # Compute direct path from hub to dst
                        try:
                            path = nx.dijkstra_path(G, hub, dst, weight='cost')
                            for i in range(len(path)-1):
                                u, v = path[i], path[i+1]
                                bc_topology.append_dst_partition_path(dst, partition, [u, v, G[u][v]])
                        except:
                            # Fallback to source-to-destination
                            if dst in src_paths:
                                path = src_paths[dst]
                                for i in range(len(path)-1):
                                    u, v = path[i], path[i+1]
                                    bc_topology.append_dst_partition_path(dst, partition, [u, v, G[u][v]])
                
                processed_hubs.add(hub)
        else:
            # Direct routing
            for dst in dsts:
                if dst in src_paths:
                    path = src_paths[dst]
                    for i in range(len(path)-1):
                        u, v = path[i], path[i+1]
                        bc_topology.append_dst_partition_path(dst, partition, [u, v, G[u][v]])
    
    # Ensure all destinations have paths for all partitions
    for dst in dsts:
        for partition in range(num_partitions):
            if not bc_topology.paths.get(dst, {}).get(str(partition), []):
                if dst in src_paths:
                    path = src_paths[dst]
                    for i in range(len(path)-1):
                        u, v = path[i], path[i+1]
                        bc_topology.append_dst_partition_path(dst, partition, [u, v, G[u][v]])
    
    return bc_topology