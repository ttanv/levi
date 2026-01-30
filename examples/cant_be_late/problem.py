"""
Can't Be Late Scheduling Problem Definition.

Uses real AWS spot traces from ADRS-Leaderboard for evaluation.
"""

import enum
import json
import math
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
    """Load trace from JSON file.

    IMPORTANT: In the raw trace data, 0 means spot IS available and 1 means
    spot is NOT available (preemption event). We invert this so that
    spot_trace[tick]=True means spot is available.
    """
    with open(trace_path, 'r') as f:
        data = json.load(f)
    gap_seconds = data['metadata']['gap_seconds']
    # Invert: trace 0 -> spot available (True), trace 1 -> spot unavailable (False)
    spot_trace = [not bool(x) for x in data['data']]
    return gap_seconds, spot_trace


def find_evaluator_traces() -> Optional[Path]:
    """Find the ADRS evaluator trace root (contains per-overhead directories).

    These are the traces used by evaluator_real30.py, organized as:
    data/real/ddl=search+task=48+overhead={overhead}/real/{env}/traces/random_start/
    """
    this_dir = Path(__file__).parent
    candidates = [
        this_dir / "traces/real",
        Path("/tmp/adrs_traces/real"),
        this_dir / "../../../ADRS-Leaderboard/problems/cant_be_late/resources/cant-be-late-simulator/data/real",
    ]
    for path in candidates:
        # Check for the per-overhead directory structure
        test_dir = path / "ddl=search+task=48+overhead=0.02"
        if test_dir.exists():
            return path.resolve()
    return None


def find_leaderboard_traces() -> Optional[Path]:
    this_dir = Path(__file__).parent
    candidates = [
        # Local traces extracted to example folder (preferred)
        this_dir / "traces/real/ping_based/random_start_time",
        # Legacy paths
        this_dir / "../../../ADRS-Leaderboard/datasets/cant_be_late/real/ping_based/random_start_time",
        Path.home() / "Documents/af/ADRS-Leaderboard/datasets/cant_be_late/real/ping_based/random_start_time",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def _select_evenly_spaced(items: list, target: int) -> list:
    """Select target items evenly spaced from items list.

    Matches ADRS evaluator_real30.py trace selection logic exactly.
    """
    if len(items) <= target:
        return items
    indices = []
    max_idx = len(items) - 1
    denom = target - 1 if target > 1 else 1
    prev = -1
    for j in range(target):
        raw = round(j * max_idx / denom)
        if raw <= prev:
            raw = prev + 1
        if raw > max_idx:
            raw = max_idx
        indices.append(raw)
        prev = raw
    return [items[i] for i in indices]


def generate_test_cases_from_real_traces(
    max_traces_per_env: int = 30,
    environments: Optional[List[str]] = None,
) -> List[SimulationConfig]:
    """Generate test cases matching ADRS evaluator_real30.py exactly.

    Uses per-overhead trace directories with evenly-spaced trace selection.
    Falls back to ping_based traces if evaluator traces not found.
    """
    # Match ADRS-Leaderboard ENV_PATHS exactly
    if environments is None:
        environments = [
            "us-west-2a_k80_8",
            "us-west-2b_k80_1",
            "us-west-2b_k80_8",
            "us-west-2a_v100_1",
            "us-west-2a_v100_8",
            "us-west-2b_v100_1",
        ]

    task_duration_hours = 48
    deadlines_hours = [52, 70]
    overheads_hours = [0.02, 0.05, 0.1]
    min_required_hours = max(deadlines_hours)

    # Try evaluator traces first (per-overhead directories)
    eval_trace_root = find_evaluator_traces()
    if eval_trace_root is not None:
        return _generate_from_evaluator_traces(
            eval_trace_root, environments, task_duration_hours,
            deadlines_hours, overheads_hours, max_traces_per_env,
            min_required_hours,
        )

    # Fallback: ping_based traces (shared across overheads)
    trace_dir = find_leaderboard_traces()
    if trace_dir is None:
        raise RuntimeError("Could not find ADRS-Leaderboard traces.")

    return _generate_from_ping_traces(
        trace_dir, environments, task_duration_hours,
        deadlines_hours, overheads_hours, max_traces_per_env,
        min_required_hours,
    )


def _generate_from_evaluator_traces(
    eval_trace_root: Path,
    environments: List[str],
    task_duration_hours: int,
    deadlines_hours: List[int],
    overheads_hours: List[float],
    max_traces_per_env: int,
    min_required_hours: float,
) -> List[SimulationConfig]:
    """Generate test cases using ADRS evaluator trace structure.

    Matches evaluator_real30.py build_trace_pool() + evaluation loop.
    """
    test_cases = []

    for overhead_h in overheads_hours:
        over_str = f"{overhead_h:.2f}"
        base_dir = eval_trace_root / f"ddl=search+task=48+overhead={over_str}" / "real"
        if not base_dir.exists():
            continue

        for env_name in environments:
            trace_dir = base_dir / env_name / "traces" / "random_start"
            if not trace_dir.exists():
                continue

            # Filter eligible traces (must cover min_required_hours)
            eligible = []
            for trace_file in sorted(trace_dir.glob("*.json")):
                try:
                    gap_seconds, spot_trace = load_trace_from_json(str(trace_file))
                except Exception:
                    continue
                total_hours = len(spot_trace) * gap_seconds / 3600.0
                if total_hours + 1e-9 < min_required_hours:
                    continue
                eligible.append((trace_file, gap_seconds, spot_trace))

            if not eligible:
                continue

            # Select evenly spaced traces (matches ADRS selection)
            selected = _select_evenly_spaced(eligible, max_traces_per_env)

            for trace_file, gap_seconds, spot_trace in selected:
                for deadline_h in deadlines_hours:
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
        raise RuntimeError("No valid evaluator traces found")

    return test_cases


def _generate_from_ping_traces(
    trace_dir: Path,
    environments: List[str],
    task_duration_hours: int,
    deadlines_hours: List[int],
    overheads_hours: List[float],
    max_traces_per_env: int,
    min_required_hours: float,
) -> List[SimulationConfig]:
    """Fallback: generate test cases from ping_based traces."""
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

            required_ticks = int(min_required_hours * 3600 / gap_seconds) + 10
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
    """Generate test cases matching ADRS evaluator_real30.py configuration.

    Uses per-overhead trace directories with 30 evenly-spaced traces per
    environment. 3 overheads * 6 envs * 30 traces * 2 deadlines = 1080 cases.
    """
    try:
        return generate_test_cases_from_real_traces(max_traces_per_env=30)
    except RuntimeError:
        print("Warning: Real traces not found, using synthetic test cases")
        return generate_synthetic_test_cases()


def simulate(strategy_step, config: SimulationConfig) -> Tuple[float, bool]:
    """Simulate matching ADRS Strategy.step() + Env flow exactly.

    Flow per tick (matches strategy.py Strategy.step()):
    1. observe(): record last cluster type in history, detect preemption
    2. Work accounting: realize PREVIOUS tick's work
    3. _step(): strategy decision (receives pre-preemption last_cluster_type)
    4. Strong guarantee override
    5. Safety override (SPOT -> NONE if unavailable)
    6. Set restart overhead based on cluster type switch
    7. env.step(): commit decision, advance tick

    Cost is computed from cluster_type_history (what was running each tick),
    matching ADRS env.accumulated_cost exactly.
    """
    SPOT_COST_PER_HOUR = 0.9731
    ON_DEMAND_COST_PER_HOUR = 3.06
    COST_MAP = {
        ClusterType.ON_DEMAND: ON_DEMAND_COST_PER_HOUR,
        ClusterType.SPOT: SPOT_COST_PER_HOUR,
        ClusterType.NONE: 0.0,
    }

    ctx = StrategyContext(config)
    remaining_restart_overhead = 0.0
    tick = 0
    task_done = False

    # Matches ADRS env.cluster_type (set by env.step() each tick)
    env_cluster_type = ClusterType.NONE
    # Matches ADRS env.cluster_type_histroy for cost calculation
    cluster_type_history: List[ClusterType] = []

    while True:
        # === ADRS while condition: check task_done ===
        if task_done:
            break

        # Trace overflow = simulation failure
        if tick >= len(config.spot_trace):
            break

        # === 1. observe() (env.py lines 46-60) ===
        has_spot = config.spot_trace[tick]
        # last_cluster_type is PRE-preemption (matches ADRS observe)
        last_cluster_type = env_cluster_type
        cluster_type_history.append(last_cluster_type)

        # Detect preemption
        if env_cluster_type == ClusterType.SPOT and not has_spot:
            env_cluster_type = ClusterType.NONE

        # === 2. Work accounting for PREVIOUS tick (strategy.py lines 112-125) ===
        if last_cluster_type == ClusterType.NONE:
            ctx.task_done_time.append(0)
        else:
            available_time = config.gap_seconds
            work = max(available_time - remaining_restart_overhead, 0)
            remaining_restart_overhead -= (available_time - work)
            if remaining_restart_overhead < 1e-3:
                remaining_restart_overhead = 0
            remaining_task = config.task_duration - sum(ctx.task_done_time)
            work = min(work, remaining_task)
            ctx.task_done_time.append(work)

        # Check if task completed during work accounting (ADRS uses 1e-8 tolerance)
        task_done = (config.task_duration - sum(ctx.task_done_time)) <= 1e-8

        # Update ctx for strategy (ADRS: env.elapsed_seconds = tick * gap)
        ctx.env.elapsed_seconds = tick * config.gap_seconds
        ctx.env.cluster_type = env_cluster_type  # post-preemption state

        # === 3. _step(): strategy decision (strategy.py line 128) ===
        # ADRS passes pre-preemption last_cluster_type to _step()
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

        # === 4. Strong guarantee (strategy.py lines 63-92) ===
        request_type = decision
        remaining_task_time = config.task_duration - sum(ctx.task_done_time)
        if remaining_task_time > 1e-3:
            remaining_time = math.floor(
                (config.deadline - ctx.env.elapsed_seconds) /
                config.gap_seconds) * config.gap_seconds
            total_task_remaining = math.ceil(
                (remaining_task_time + config.restart_overhead) /
                config.gap_seconds) * config.gap_seconds
            if total_task_remaining >= remaining_time:
                if last_cluster_type == ClusterType.SPOT and remaining_restart_overhead < 1e-3:
                    request_type = ClusterType.SPOT
                else:
                    request_type = ClusterType.ON_DEMAND

        # === 5. Safety override (strategy.py lines 142-148) ===
        if request_type == ClusterType.SPOT and not has_spot:
            request_type = ClusterType.NONE

        # === 6. Restart overhead for switching (strategy.py lines 171-181) ===
        current_cluster_type = last_cluster_type
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            current_cluster_type = ClusterType.NONE
        if current_cluster_type != request_type and request_type != ClusterType.NONE:
            remaining_restart_overhead = config.restart_overhead

        # === 7. env.step(): commit decision, advance tick ===
        env_cluster_type = request_type
        tick += 1

    # === Final step (simulate.py lines 102-104) ===
    # ADRS does one more strategy.step() after loop exits (to realize last
    # tick's work via observe). The observe appends the last env_cluster_type
    # to history. Then env.step(NONE).
    if task_done and tick < len(config.spot_trace):
        cluster_type_history.append(env_cluster_type)
        # Work accounting: remaining_task <= 0, so 0 work appended
        # env.step(NONE): doesn't add to history (no more observe)

    # === Cost calculation (env.py accumulated_cost) ===
    hours = config.gap_seconds / 3600
    total_cost = sum(COST_MAP[ct] * hours for ct in cluster_type_history)

    met_deadline = (config.task_duration - sum(ctx.task_done_time)) <= 1e-8
    return total_cost, met_deadline


PROBLEM_DESCRIPTION = """
# Can't Be Late: Cloud Instance Scheduling

## Problem
Schedule compute tasks on cloud infrastructure to minimize cost while meeting deadlines.

## Instance Types
- **SPOT**: Cheap but can be preempted at any time, causing job restart
- **ON_DEMAND**: Guaranteed availability but expensive
- **NONE**: No instance running (no cost, no progress)

## Key Concepts
- `task_duration`: Total compute time needed (48 hours typical)
- `deadline`: When task must complete
- `restart_overhead`: Time penalty when switching instances or after preemption
- `gap_seconds`: Time step size (600 seconds = 10 minutes typical)

## Strategy Interface
At each time step, your strategy receives:
- `ctx`: Context with task info and current state
- `last_cluster_type`: What instance was running last tick (pre-preemption: still SPOT if just preempted)
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
"""

FUNCTION_SIGNATURE = """
import enum

class ClusterType(str, enum.Enum):
    NONE = "NONE"
    SPOT = "SPOT"
    ON_DEMAND = "ON_DEMAND"

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
        last_cluster_type: Previous instance type, pre-preemption (SPOT if just preempted)
        has_spot: Whether spot is available this tick

    Returns:
        ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
    '''
    pass
"""

SEED_PROGRAM = '''import enum

class ClusterType(str, enum.Enum):
    NONE = "NONE"
    SPOT = "SPOT"
    ON_DEMAND = "ON_DEMAND"

def strategy_step(ctx, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
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

## Required Code Structure
Your code MUST define ClusterType and the strategy function:
```python
import enum

class ClusterType(str, enum.Enum):
    NONE = "NONE"
    SPOT = "SPOT"
    ON_DEMAND = "ON_DEMAND"

def strategy_step(ctx, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
    pass
```

## Your Task: ALGORITHMIC DIVERSITY

Design a **FUNDAMENTALLY DIFFERENT ALGORITHM** than existing seeds.
**Deadline miss = score 0.** Ensure safety margins before optimizing cost.

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
- ClusterType not defined (code MUST define ClusterType enum before using it)
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
    Evaluate scheduling strategy using ADRS-Leaderboard scoring methodology.

    Scoring matches ADRS run_evaluator_real30.py:
    1. Run all simulations and collect costs
    2. If ANY simulation fails (deadline miss), return score 0
    3. Compute average cost across all scenarios
    4. Normalize: score = (od_anchor - avg_cost) / (od_anchor - spot_anchor) * 100
       (unrounded for finer evolution granularity; ADRS rounds to integer)

    Where:
    - od_anchor = task_hours * ON_DEMAND_COST = 48 * 3.06 = 146.88
    - spot_anchor = task_hours * SPOT_COST = 48 * 0.9731 = 46.7088
    """
    try:
        SPOT_COST_PER_HOUR = 0.9731
        ON_DEMAND_COST_PER_HOUR = 3.06

        all_costs = []
        tight_deadline_costs = []
        loose_deadline_costs = []

        for config in inputs:
            cost, met_deadline = simulate(strategy_step, config)

            # ADRS behavior: if ANY simulation fails, entire evaluation fails
            if not met_deadline:
                return {
                    "score": 0,
                    "num_tests": len(inputs),
                    "tight_deadline_score": 0.0,
                    "loose_deadline_score": 0.0,
                    "avg_cost": float('inf'),
                    "error": "Deadline missed",
                }

            all_costs.append(cost)

            if config.deadline_hours == 52:
                tight_deadline_costs.append(cost)
            if config.deadline_hours == 70:
                loose_deadline_costs.append(cost)

        if not all_costs:
            return {"error": "No valid test cases"}

        # Compute average cost (matches ADRS avg_cost calculation)
        avg_cost = sum(all_costs) / len(all_costs)

        # Compute anchors (same for all scenarios since task_duration is constant)
        task_hours = inputs[0].task_duration / 3600  # 48 hours
        od_anchor = task_hours * ON_DEMAND_COST_PER_HOUR  # 146.88
        spot_anchor = task_hours * SPOT_COST_PER_HOUR  # 46.7088

        # Normalize and compute score
        denom = od_anchor - spot_anchor
        if denom <= 1e-9:
            score = 0.0
        else:
            norm = (od_anchor - avg_cost) / denom
            norm = max(0.0, min(1.0, norm))
            score = norm * 100  # Unrounded for finer granularity

        # Compute sub-scores for tight/loose deadlines
        def compute_subscore(costs):
            if not costs:
                return 0.0
            avg = sum(costs) / len(costs)
            norm = (od_anchor - avg) / denom if denom > 1e-9 else 0.0
            norm = max(0.0, min(1.0, norm))
            return norm * 100  # Unrounded

        return {
            "score": score,
            "num_tests": len(all_costs),
            "tight_deadline_score": compute_subscore(tight_deadline_costs),
            "loose_deadline_score": compute_subscore(loose_deadline_costs),
            "avg_cost": avg_cost,
            "od_anchor": od_anchor,
            "spot_anchor": spot_anchor,
        }

    except Exception as e:
        return {"error": str(e)}
