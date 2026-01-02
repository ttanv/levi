"""
Prompts for Cloudcast Broadcast Optimization evolution.
"""

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

def search_algorithm(src: str, dsts: List[str], G: nx.DiGraph, num_partitions: int) -> BroadCastTopology:
    '''
    Find optimal broadcast topology from src to all destinations.

    Args:
        src: Source node (cloud region)
        dsts: List of destination nodes
        G: NetworkX DiGraph with 'cost' and 'throughput' edge attributes
        num_partitions: Number of data partitions

    Returns:
        BroadCastTopology with paths for each (dst, partition) pair
    '''
    pass
"""

SEED_PROGRAM = '''# EVOLVE-BLOCK-START
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


# EVOLVE-BLOCK-END
'''

# No inspiration seeds - matching OpenEvolve which only has a single seed program
SEED_INSPIRATIONS = []

DIVERSITY_SEED_PROMPT = """
# Cloudcast Broadcast Optimization

## Problem
Optimize broadcast topology for multi-cloud data distribution across AWS, Azure, and GCP.

## Key Concepts
- Graph G has edge attributes: `cost` ($/GB) and `throughput` (Gbps)
- BroadCastTopology stores paths for each (destination, partition) pair
- Data is partitioned into chunks that can take different paths

## Objective
Minimize total transfer cost ($/GB) across 5 network configurations.

## APIs
- `G[src][dst]['cost']` - Cost per GB
- `G[src][dst]['throughput']` - Throughput in Gbps
- `nx.dijkstra_path(G, src, dst, weight='cost')` - Shortest path by cost
- `BroadCastTopology(src, dsts, num_partitions)` - Create topology
- `bc_topology.append_dst_partition_path(dst, partition, [src, tgt, edge_data])` - Add segment

## Function Signature
```python
import networkx as nx
from typing import List

def search_algorithm(src: str, dsts: List[str], G: nx.DiGraph, num_partitions: int) -> BroadCastTopology:
    '''Find optimal broadcast topology from src to all destinations.'''
    pass
```

## Your Task
Generate a solution with DIFFERENT BEHAVIORAL CHARACTERISTICS than the existing seeds.

**CRITICAL: BEHAVIORAL DIVERSITY IS ESSENTIAL.**

Focus on creating solutions that exhibit different runtime behaviors:
- Different path selection strategies (shortest cost vs balanced throughput)
- Different handling of partitions (same path vs diversified paths)
- Different exploration of the graph (greedy vs comprehensive)
- Different tradeoffs between cost and throughput optimization

The goal is behavioral variety in the population, not just different code.

## Existing Seeds (aim for different behavioral characteristics):
{existing_seeds}

## Instructions
1. Review the existing seeds and consider their likely runtime behavior
2. Design a solution that would behave differently (e.g., different cost/throughput tradeoff)
3. Implement it as a complete, working solution
4. Output ONLY the complete Python code in a ```python block
"""

META_ADVISOR_PROMPT = """You are a meta-advisor for an evolutionary code optimization system.

## Your Role
Analyze evolution metrics and provide strategic guidance. Your advice gets injected into LLM prompts to steer the next generation of solutions.

## Problem Context: CLOUDCAST BROADCAST OPTIMIZATION
- Minimize transfer cost across multi-cloud networks (AWS, Azure, GCP)
- 5 configurations are evaluated: intra_aws, intra_azure, intra_gcp, inter_agz, inter_gaz2
- Score is 0-100 based on cost reduction from baseline ($1199) toward optimal ($626)

## What You're Given
- **Period Metrics**: Acceptance/rejection/error rates from recent evaluations
- **Error Messages**: Specific failure patterns to avoid
- **Previous Advice**: What you recommended last time
- **Best Solution**: Current top performer's code
- **Progress**: Budget consumption percentage

## Your Task: Write Strategic Advice (400-500 words)

### 1. Reflect on Previous Advice
- Look at the metrics: did your last advice help or hurt?
- If acceptance rate improved → reinforce what worked
- If errors increased → explicitly retract problematic suggestions
- If no change → your advice may have been too vague, be more specific

### 2. Interpret the Metrics
Diagnose what the numbers tell you:
- **High rejection, low error**: Valid code but not improving → need MORE diversity, structural changes
- **High error rate**: Bugs in generated code → identify patterns from error messages, warn against them
- **Low acceptance + plateau**: Archive saturated → need fundamentally different algorithmic approaches
- **Good acceptance rate**: Current direction working → encourage deeper exploration of similar ideas

### 3. Analyze the Best Solution
Look at the provided code:
- What is its core algorithmic strategy?
- What are its likely weaknesses or blind spots?
- What aspects of the problem might it be ignoring?
- Suggest exploring what it DOESN'T do

### 4. Give Actionable Direction
Be SPECIFIC about what to try differently. Bad advice: "try something different." Good advice:
- Identify what the best solution focuses on, then suggest alternatives
- If current solutions ignore some aspect, recommend exploring it
- Point to specific structural changes rather than parameter tweaks

### 5. Warn Against Failure Patterns
Based on error messages, give explicit warnings:
- Quote specific error patterns and explain how to avoid them
- If timeout errors: suggest ways to reduce computational complexity
- If assertion errors: highlight what invariants must be maintained

## Critical Rules
- Each LLM sees a DIFFERENT parent solution (not the best one)
- Tell them to STRUCTURALLY modify their given parent
- Don't tell them to copy the best solution
- Push for paradigm changes, not parameter tweaks
- Your advice should evolve based on what's working/failing

## Output Format
Structure your advice clearly:
1. **What's Working** (based on metrics)
2. **What to Avoid** (based on errors)
3. **Strategic Direction** (what to try next)
4. **Specific Suggestions** (2-3 concrete ideas derived from analysis)

---

{metrics_data}

Provide your strategic advice:"""
