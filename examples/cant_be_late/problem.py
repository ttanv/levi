"""
Can't Be Late Scheduling Problem Definition.

Uses real AWS spot traces from ADRS-Leaderboard for evaluation.
"""

import enum
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional


class ClusterType(str, enum.Enum):
    NONE = "NONE"
    SPOT = "SPOT"
    ON_DEMAND = "ON_DEMAND"


@dataclass
class SimulationConfig:
    task_duration: float
    deadline: float
    restart_overhead: float
    gap_seconds: float
    spot_trace: List[bool]
    deadline_hours: float = 0.0
    overhead_hours: float = 0.0


@dataclass
class SimulationState:
    elapsed_seconds: float
    gap_seconds: float
    cluster_type: ClusterType


class StrategyContext:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.task_duration = config.task_duration
        self.deadline = config.deadline
        self.restart_overhead = config.restart_overhead
        self.task_done_time: List[float] = []
        self.env = SimulationState(
            elapsed_seconds=0.0,
            gap_seconds=config.gap_seconds,
            cluster_type=ClusterType.NONE
        )


def load_trace_from_json(trace_path: str) -> Tuple[float, List[bool]]:
    with open(trace_path, 'r') as f:
        data = json.load(f)
    gap_seconds = data['metadata']['gap_seconds']
    spot_trace = [bool(x) for x in data['data']]
    return gap_seconds, spot_trace


def find_leaderboard_traces() -> Optional[Path]:
    this_dir = Path(__file__).parent
    candidates = [
        this_dir / "../../../ADRS-Leaderboard/datasets/cant_be_late/real/ping_based/random_start_time",
        Path.home() / "Documents/af/ADRS-Leaderboard/datasets/cant_be_late/real/ping_based/random_start_time",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def generate_test_cases_from_real_traces(
    max_traces_per_env: int = 5,
    environments: Optional[List[str]] = None,
) -> List[SimulationConfig]:
    trace_dir = find_leaderboard_traces()
    if trace_dir is None:
        raise RuntimeError("Could not find ADRS-Leaderboard traces.")

    if environments is None:
        environments = ["us-west-2a_k80_1", "us-west-2a_v100_1", "us-west-2b_k80_1"]

    task_duration_hours = 48
    deadlines_hours = [52, 70]
    overheads_hours = [0.02, 0.05, 0.1]

    test_cases = []

    for env_name in environments:
        env_dir = trace_dir / env_name
        if not env_dir.exists():
            continue

        trace_files = sorted(env_dir.glob("*.json"))[:max_traces_per_env]

        for trace_file in trace_files:
            try:
                gap_seconds, spot_trace = load_trace_from_json(str(trace_file))
            except Exception:
                continue

            max_deadline_hours = max(deadlines_hours)
            required_ticks = int(max_deadline_hours * 3600 / gap_seconds) + 10
            if len(spot_trace) < required_ticks:
                continue

            for deadline_h in deadlines_hours:
                for overhead_h in overheads_hours:
                    test_cases.append(SimulationConfig(
                        task_duration=task_duration_hours * 3600,
                        deadline=deadline_h * 3600,
                        restart_overhead=overhead_h * 3600,
                        gap_seconds=gap_seconds,
                        spot_trace=spot_trace,
                        deadline_hours=deadline_h,
                        overhead_hours=overhead_h,
                    ))

    if not test_cases:
        raise RuntimeError(f"No valid traces found in {trace_dir}")

    return test_cases


def generate_synthetic_test_cases(num_cases: int = 30, seed: int = 42) -> List[SimulationConfig]:
    random.seed(seed)

    task_duration_hours = 48
    gap_hours = 1.0
    deadlines_hours = [52, 70]
    overheads_hours = [0.02, 0.05, 0.1]

    test_cases = []
    for i in range(num_cases):
        deadline_h = random.choice(deadlines_hours)
        overhead_h = random.choice(overheads_hours)
        preempt_rate = random.choice([0.02, 0.05, 0.10])

        n_ticks = int(deadline_h / gap_hours) + 10

        trace = []
        spot_available = True
        for _ in range(n_ticks):
            if spot_available and random.random() < preempt_rate:
                spot_available = False
            elif not spot_available and random.random() < 0.3:
                spot_available = True
            trace.append(spot_available)

        test_cases.append(SimulationConfig(
            task_duration=task_duration_hours * 3600,
            deadline=deadline_h * 3600,
            restart_overhead=overhead_h * 3600,
            gap_seconds=gap_hours * 3600,
            spot_trace=trace,
            deadline_hours=deadline_h,
            overhead_hours=overhead_h,
        ))

    return test_cases


def generate_test_cases() -> List[SimulationConfig]:
    try:
        return generate_test_cases_from_real_traces(max_traces_per_env=5)
    except RuntimeError:
        print("Warning: Real traces not found, using synthetic test cases")
        return generate_synthetic_test_cases()


def simulate(strategy_step, config: SimulationConfig) -> Tuple[float, bool]:
    SPOT_COST_PER_HOUR = 0.97
    ON_DEMAND_COST_PER_HOUR = 3.06

    ctx = StrategyContext(config)
    total_cost = 0.0
    remaining_restart_overhead = 0.0
    tick = 0
    last_cluster_type = ClusterType.NONE

    while ctx.env.elapsed_seconds < config.deadline:
        has_spot = config.spot_trace[tick] if tick < len(config.spot_trace) else True

        if last_cluster_type == ClusterType.SPOT and not has_spot:
            last_cluster_type = ClusterType.NONE
            remaining_restart_overhead = config.restart_overhead

        try:
            decision = strategy_step(ctx, last_cluster_type, has_spot)
        except Exception:
            return float('inf'), False

        if not isinstance(decision, ClusterType):
            if isinstance(decision, str):
                try:
                    decision = ClusterType(decision)
                except ValueError:
                    return float('inf'), False
            else:
                return float('inf'), False

        if decision == ClusterType.SPOT and not has_spot:
            decision = ClusterType.NONE

        work_done = 0.0
        if decision != ClusterType.NONE:
            available_time = config.gap_seconds
            if last_cluster_type != decision or last_cluster_type == ClusterType.NONE:
                remaining_restart_overhead = config.restart_overhead

            work_done = max(0, available_time - remaining_restart_overhead)
            remaining_restart_overhead = max(0, remaining_restart_overhead - available_time)
            remaining_work = config.task_duration - sum(ctx.task_done_time)
            work_done = min(work_done, remaining_work)

        ctx.task_done_time.append(work_done)

        hours = config.gap_seconds / 3600
        if decision == ClusterType.SPOT:
            total_cost += SPOT_COST_PER_HOUR * hours
        elif decision == ClusterType.ON_DEMAND:
            total_cost += ON_DEMAND_COST_PER_HOUR * hours

        last_cluster_type = decision
        ctx.env.elapsed_seconds += config.gap_seconds
        ctx.env.cluster_type = decision
        tick += 1

        if sum(ctx.task_done_time) >= config.task_duration - 1e-3:
            break

    met_deadline = sum(ctx.task_done_time) >= config.task_duration - 1e-3
    return total_cost, met_deadline


PROBLEM_DESCRIPTION = """
# Can't Be Late: Cloud Instance Scheduling

## Problem
Schedule compute tasks on cloud infrastructure to minimize cost while meeting deadlines.

## Instance Types
- **SPOT**: Cheap (~$0.97/hr) but can be preempted at any time, causing job restart
- **ON_DEMAND**: Expensive (~$3.06/hr) but guaranteed availability
- **NONE**: No instance running (no cost, no progress)

## Key Concepts
- `task_duration`: Total compute time needed (48 hours typical)
- `deadline`: When task must complete
- `restart_overhead`: Time penalty when switching instances or after preemption
- `gap_seconds`: Time step size (600 seconds = 10 minutes typical)

## Strategy Interface
At each time step, your strategy receives:
- `ctx`: Context with task info and current state
- `last_cluster_type`: What instance was running (NONE, SPOT, or ON_DEMAND)
- `has_spot`: Whether spot instance is available this tick

Return: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE

## Available Information
- `ctx.task_duration`: Total work needed (seconds)
- `ctx.deadline`: Deadline time (seconds)
- `ctx.restart_overhead`: Time penalty on restart (seconds)
- `ctx.task_done_time`: List of work done each tick (sum for total progress)
- `ctx.env.elapsed_seconds`: Current time
- `ctx.env.gap_seconds`: Time step size
- `ctx.env.cluster_type`: Current cluster type

## Scoring
- Score 0-100 based on cost relative to baselines
- 0 = Cost of running fully on-demand (worst)
- 100 = Cost of running fully on spot without preemptions (best)
- Deadline miss = score 0

## You can import: random, math, collections
"""

FUNCTION_SIGNATURE = """
def strategy_step(ctx, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
    '''
    Decide which instance type to use for the next time step.

    Args:
        ctx: Strategy context with:
            - ctx.task_duration: Total work needed (seconds)
            - ctx.deadline: Deadline (seconds)
            - ctx.restart_overhead: Restart penalty (seconds)
            - ctx.task_done_time: List of work done per tick
            - ctx.env.elapsed_seconds: Current time
            - ctx.env.gap_seconds: Time step size
        last_cluster_type: Previous instance type (ClusterType.NONE/SPOT/ON_DEMAND)
        has_spot: Whether spot is available this tick

    Returns:
        ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
    '''
    pass
"""

SEED_PROGRAM = '''def strategy_step(ctx, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
    """Greedy safety strategy: use spot until deadline pressure forces on-demand."""
    import math

    gap = ctx.env.gap_seconds
    work_left = ctx.task_duration - sum(ctx.task_done_time)

    if work_left <= 1e-9:
        return ClusterType.NONE

    remaining_time = ctx.deadline - ctx.env.elapsed_seconds
    left_ticks = max(0, math.floor(remaining_time / gap))

    need_1_restart = math.ceil((work_left + ctx.restart_overhead) / gap)
    need_2_restart = math.ceil((work_left + 2 * ctx.restart_overhead) / gap)

    if need_1_restart >= left_ticks:
        return ClusterType.ON_DEMAND

    if need_2_restart >= left_ticks:
        if last_cluster_type == ClusterType.SPOT and has_spot:
            return ClusterType.SPOT
        return ClusterType.ON_DEMAND

    return ClusterType.SPOT if has_spot else ClusterType.NONE
'''

SEED_INSPIRATIONS = []

DIVERSITY_SEED_PROMPT = """
# Can't Be Late: Cloud Scheduling

## Problem
Schedule tasks on spot (cheap, preemptable) vs on-demand (expensive, reliable) instances.

## Input
- `ctx`: Task context (duration, deadline, restart_overhead, task_done_time, env)
- `last_cluster_type`: Previous instance (NONE/SPOT/ON_DEMAND)
- `has_spot`: Spot availability this tick

## Output
Return: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE

## Function
```python
def strategy_step(ctx, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
    pass
```

## Your Task: ALGORITHMIC DIVERSITY

Design a **FUNDAMENTALLY DIFFERENT ALGORITHM** than existing seeds.

## Existing Seeds:
{existing_seeds}

## Output
Output ONLY complete Python code in a ```python block.
"""

META_ADVISOR_PROMPT = """You are a lessons-learned advisor for an evolutionary code optimization system.

## Your Role
Analyze FAILURES from recent evaluations. Your lessons get injected into LLM prompts.

## Focus on Failure Prevention
1. Identify error patterns
2. Explain root causes
3. Give specific fixes

## Common Issues
- Returning wrong type (must return ClusterType enum)
- Missing deadline (need to switch to on-demand earlier)
- Infinite loops or exceptions

## Output Format
Keep it SHORT and DIRECT:

**Avoid These Errors:**
- [Error pattern]: [How to fix]

---

{metrics_data}

Provide your lessons (150-200 words max):"""


INPUTS = generate_test_cases()


def score_fn(strategy_step, inputs: List[SimulationConfig]):
    """
    Evaluate scheduling strategy. Single pass computes overall score
    plus scenario-based behavioral dimensions (tight_deadline_score, high_overhead_score).
    """
    try:
        SPOT_COST_PER_HOUR = 0.97
        ON_DEMAND_COST_PER_HOUR = 3.06

        all_scores = []
        tight_deadline_scores = []
        high_overhead_scores = []

        for config in inputs:
            cost, met_deadline = simulate(strategy_step, config)

            if not met_deadline:
                test_score = 0.0
            else:
                task_hours = config.task_duration / 3600
                od_cost = task_hours * ON_DEMAND_COST_PER_HOUR
                spot_cost = task_hours * SPOT_COST_PER_HOUR

                if od_cost > spot_cost:
                    raw = (od_cost - cost) / (od_cost - spot_cost)
                    test_score = max(0.0, min(1.0, raw)) * 100
                else:
                    test_score = 100.0 if cost <= spot_cost else 0.0

            all_scores.append(test_score)

            if config.deadline_hours == 52:
                tight_deadline_scores.append(test_score)
            if config.overhead_hours == 0.1:
                high_overhead_scores.append(test_score)

        if not all_scores:
            return {"error": "No valid test cases"}

        return {
            "score": sum(all_scores) / len(all_scores),
            "num_tests": len(all_scores),
            "tight_deadline_score": (
                sum(tight_deadline_scores) / len(tight_deadline_scores)
                if tight_deadline_scores else 0.0
            ),
            "high_overhead_score": (
                sum(high_overhead_scores) / len(high_overhead_scores)
                if high_overhead_scores else 0.0
            ),
        }

    except Exception as e:
        return {"error": str(e)}
