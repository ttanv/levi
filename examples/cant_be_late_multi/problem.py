"""
Can't Be Late Multi-Region Scheduling Problem Definition.

Uses real AWS spot traces from ADRS-Leaderboard for multi-region evaluation.
Strategies can switch between regions to find spot availability.
"""

import enum
import json
import math
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


class ClusterType(str, enum.Enum):
    NONE = "NONE"
    SPOT = "SPOT"
    ON_DEMAND = "ON_DEMAND"


@dataclass
class MultiRegionSimulationConfig:
    scenario_name: str
    task_duration: float
    deadline: float
    restart_overhead: float
    gap_seconds: float
    region_traces: List[List[bool]]  # Per-region spot traces
    region_spot_prices: List[float] = field(default_factory=list)  # Per-region spot $/hr
    region_od_price: float = 3.06  # On-demand $/hr (same for all v100_1)
    trace_id: str = ""


@dataclass
class MultiRegionSimulationState:
    elapsed_seconds: float
    gap_seconds: float
    cluster_type: ClusterType
    _region_traces: List[List[bool]] = field(default_factory=list, repr=False)
    _current_region: int = 0

    def get_current_region(self) -> int:
        return self._current_region

    def switch_region(self, idx: int) -> bool:
        if 0 <= idx < len(self._region_traces):
            self._current_region = idx
            return True
        return False

    def get_num_regions(self) -> int:
        return len(self._region_traces)

    def get_all_regions_spot_available(self) -> List[bool]:
        """Return spot availability for all regions at the current tick."""
        tick = int(self.elapsed_seconds / self.gap_seconds)
        result = []
        for traces in self._region_traces:
            if tick < len(traces):
                result.append(traces[tick])
            else:
                result.append(False)
        return result


class StrategyContext:
    def __init__(self, config: MultiRegionSimulationConfig):
        self.config = config
        self.task_duration = config.task_duration
        self.deadline = config.deadline
        self.restart_overhead = config.restart_overhead
        self.task_done_time: List[float] = []
        self.env = MultiRegionSimulationState(
            elapsed_seconds=0.0,
            gap_seconds=config.gap_seconds,
            cluster_type=ClusterType.NONE,
            _region_traces=config.region_traces,
            _current_region=0,
        )


# --- Fixed parameters matching ADRS evaluator.py ---
TASK_DURATION_HOURS = 48
DEADLINE_HOURS = 52
RESTART_OVERHEAD_HOURS = 0.2

SPOT_COST_PER_HOUR = 0.9731
ON_DEMAND_COST_PER_HOUR = 3.06

# --- Test scenarios matching ADRS evaluator.py FULL_TEST_SCENARIOS exactly ---
FULL_TEST_SCENARIOS = [
    {"name": "2_zones_same_region", "regions": ["us-east-1a_v100_1", "us-east-1c_v100_1"], "traces": [f"{i}.json" for i in range(8)]},
    {"name": "2_regions_east_west", "regions": ["us-east-2a_v100_1", "us-west-2a_v100_1"], "traces": [f"{i}.json" for i in range(8)]},
    {"name": "3_regions_diverse", "regions": ["us-east-1a_v100_1", "us-east-2b_v100_1", "us-west-2c_v100_1"], "traces": [f"{i}.json" for i in range(6)]},
    {"name": "3_zones_same_region", "regions": ["us-east-1a_v100_1", "us-east-1c_v100_1", "us-east-1d_v100_1"], "traces": [f"{i}.json" for i in range(6)]},
    {"name": "5_regions_high_diversity", "regions": ["us-east-1a_v100_1", "us-east-1f_v100_1", "us-west-2a_v100_1", "us-west-2b_v100_1", "us-east-2b_v100_1"], "traces": [f"{i}.json" for i in range(4)]},
    {"name": "all_9_regions", "regions": ["us-east-2a_v100_1", "us-west-2c_v100_1", "us-east-1d_v100_1", "us-east-2b_v100_1", "us-west-2a_v100_1", "us-east-1f_v100_1", "us-east-1a_v100_1", "us-west-2b_v100_1", "us-east-1c_v100_1"], "traces": [f"{i}.json" for i in range(2)]}
]


def load_trace_from_json(trace_path: str) -> Tuple[float, List[bool], float]:
    """Load trace from JSON file.

    IMPORTANT: In the raw trace data, 0 means spot IS available and 1 means
    spot is NOT available (preemption event). We invert this so that
    trace[tick]=True means spot is available.

    Returns (gap_seconds, spot_trace, spot_price_per_hour).
    Spot price comes from the trace's prices array (index 0).
    Falls back to base_price / cost_k = 0.9731 if no prices.
    """
    with open(trace_path, 'r') as f:
        data = json.load(f)
    gap_seconds = data['metadata']['gap_seconds']
    spot_trace = [not bool(x) for x in data['data']]

    # Extract spot price matching ADRS TraceEnv logic:
    # If trace has 'prices' array, use prices[0]
    # Otherwise, fallback to base_price / COST_K = 3.06 / (3.06/0.9731) = 0.9731
    prices = data.get('prices', None)
    if prices and prices[0] is not None:
        spot_price = float(prices[0])
    else:
        spot_price = SPOT_COST_PER_HOUR  # 0.9731 fallback

    return gap_seconds, spot_trace, spot_price


def find_trace_dir() -> Optional[Path]:
    """Find the converted_multi_region_aligned trace directory."""
    this_dir = Path(__file__).parent
    candidates = [
        this_dir / "traces" / "converted_multi_region_aligned",
        this_dir / "../../../ADRS-Leaderboard/problems/cant_be_late_multi/resources/cant-be-late-simulator/data/converted_multi_region_aligned",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def extract_traces_if_needed() -> Optional[Path]:
    """Auto-extract traces from tarball if not already extracted."""
    this_dir = Path(__file__).parent
    local_trace_dir = this_dir / "traces" / "converted_multi_region_aligned"
    if local_trace_dir.exists():
        return local_trace_dir.resolve()

    tarball_paths = [
        this_dir / "../../../ADRS-Leaderboard/problems/cant_be_late_multi/resources/real_traces.tar.gz",
    ]
    for tarball in tarball_paths:
        if tarball.exists():
            print(f"Extracting traces from {tarball}...")
            extract_to = this_dir / "traces"
            extract_to.mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(tarball), 'r:gz') as tar:
                tar.extractall(path=str(extract_to))
            if local_trace_dir.exists():
                return local_trace_dir.resolve()
    return None


def generate_test_cases() -> List[MultiRegionSimulationConfig]:
    """Generate test cases matching ADRS evaluator.py FULL_TEST_SCENARIOS."""
    trace_dir = find_trace_dir()
    if trace_dir is None:
        trace_dir = extract_traces_if_needed()
    if trace_dir is None:
        raise RuntimeError(
            "Could not find multi-region traces. Expected at "
            "traces/converted_multi_region_aligned/ or ADRS-Leaderboard path."
        )

    task_duration = TASK_DURATION_HOURS * 3600
    deadline = DEADLINE_HOURS * 3600
    restart_overhead = RESTART_OVERHEAD_HOURS * 3600

    test_cases = []
    for scenario in FULL_TEST_SCENARIOS:
        scenario_name = scenario["name"]
        regions = scenario["regions"]

        for trace_file_name in scenario["traces"]:
            # Load traces for all regions for this trace index
            region_traces = []
            region_spot_prices = []
            gap_seconds = None
            valid = True

            for region in regions:
                trace_path = trace_dir / region / trace_file_name
                if not trace_path.exists():
                    print(f"Warning: Missing trace {trace_path}")
                    valid = False
                    break
                gs, spot_trace, spot_price = load_trace_from_json(str(trace_path))
                if gap_seconds is None:
                    gap_seconds = gs
                else:
                    assert abs(gap_seconds - gs) < 1e-6, \
                        f"Mismatched gap_seconds across regions: {gap_seconds} vs {gs}"
                region_traces.append(spot_trace)
                region_spot_prices.append(spot_price)

            if not valid or gap_seconds is None:
                continue

            test_cases.append(MultiRegionSimulationConfig(
                scenario_name=scenario_name,
                task_duration=task_duration,
                deadline=deadline,
                restart_overhead=restart_overhead,
                gap_seconds=gap_seconds,
                region_traces=region_traces,
                region_spot_prices=region_spot_prices,
                trace_id=trace_file_name,
            ))

    if not test_cases:
        raise RuntimeError("No valid multi-region test cases generated")

    return test_cases


def simulate(strategy_step, config: MultiRegionSimulationConfig) -> Tuple[float, bool]:
    """Simulate matching ADRS Strategy.step() + MultiTraceEnv flow.

    Flow per tick:
    1. observe(): Check spot in current region, detect preemption
    2. Work accounting: Realize PREVIOUS tick's work
    3. _step(): Strategy decision (receives pre-preemption last_cluster_type)
       - Strategy may call ctx.env.switch_region() here
    4. Read back current_region after strategy returns (captures any switch)
    5. Strong guarantee override
    6. Safety override: Check spot in CURRENT region (post-switch)
    7. Restart overhead on cluster type change
    8. Advance tick

    Cost calculation matches ADRS MultiTraceEnv.accumulated_cost:
    Uses the FINAL current region's spot price for ALL historical ticks.
    """
    ctx = StrategyContext(config)
    remaining_restart_overhead = 0.0
    tick = 0
    task_done = False

    env_cluster_type = ClusterType.NONE
    cluster_type_history: List[ClusterType] = []
    current_region = 0

    # Min trace length across all regions
    max_ticks = min(len(t) for t in config.region_traces)

    while True:
        if task_done:
            break

        if tick >= max_ticks:
            break

        # === 1. observe() ===
        # has_spot is for CURRENT region BEFORE any switch
        has_spot = config.region_traces[current_region][tick]
        last_cluster_type = env_cluster_type
        cluster_type_history.append(last_cluster_type)

        # Detect preemption in current region
        if env_cluster_type == ClusterType.SPOT and not has_spot:
            env_cluster_type = ClusterType.NONE

        # === 2. Work accounting for PREVIOUS tick ===
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

        # Check if task completed
        task_done = (config.task_duration - sum(ctx.task_done_time)) <= 1e-8

        # Update ctx for strategy
        ctx.env.elapsed_seconds = tick * config.gap_seconds
        ctx.env.cluster_type = env_cluster_type
        ctx.env._current_region = current_region

        # === 3. _step(): Strategy decision ===
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

        # === 4. Read back current_region after strategy (captures any switch) ===
        current_region = ctx.env._current_region

        # === 5. Strong guarantee ===
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

        # === 6. Safety override: check spot in CURRENT region (post-switch) ===
        if request_type == ClusterType.SPOT and not config.region_traces[current_region][tick]:
            request_type = ClusterType.NONE

        # === 7. Restart overhead for switching ===
        current_cluster_type = last_cluster_type
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            current_cluster_type = ClusterType.NONE
        if current_cluster_type != request_type and request_type != ClusterType.NONE:
            remaining_restart_overhead = config.restart_overhead

        # === 8. Commit decision, advance tick ===
        env_cluster_type = request_type
        tick += 1

    # === Final step: realize last tick's work ===
    if task_done and tick < max_ticks:
        cluster_type_history.append(env_cluster_type)

    # === Cost calculation (matches ADRS MultiTraceEnv.accumulated_cost) ===
    # ADRS uses get_constant_cost_map() from the FINAL current region's TraceEnv.
    # This applies that region's spot price to ALL historical ticks.
    if config.region_spot_prices:
        spot_price = config.region_spot_prices[current_region]
        od_price = config.region_od_price
    else:
        spot_price = SPOT_COST_PER_HOUR
        od_price = ON_DEMAND_COST_PER_HOUR
    cost_map = {
        ClusterType.ON_DEMAND: od_price,
        ClusterType.SPOT: spot_price,
        ClusterType.NONE: 0.0,
    }
    hours = config.gap_seconds / 3600
    total_cost = sum(cost_map[ct] * hours for ct in cluster_type_history)

    met_deadline = (config.task_duration - sum(ctx.task_done_time)) <= 1e-8
    return total_cost, met_deadline


PROBLEM_DESCRIPTION = """
# Can't Be Late: Multi-Region Cloud Instance Scheduling

## Problem
Schedule compute tasks on cloud infrastructure across MULTIPLE regions to minimize cost
while meeting deadlines. You can switch between regions to find spot availability.

## Instance Types
- **SPOT**: Cheap but can be preempted at any time, causing job restart
- **ON_DEMAND**: Guaranteed availability but expensive
- **NONE**: No instance running (no cost, no progress)

## Key Concepts
- `task_duration`: Total compute time needed (48 hours)
- `deadline`: When task must complete (52 hours)
- `restart_overhead`: Time penalty when switching instances or after preemption (0.2 hours)
- `gap_seconds`: Time step size
- Multiple regions with independent spot availability

## Multi-Region API
The strategy can query and switch between regions:
- `ctx.env.get_num_regions()` — Number of available regions (2-9)
- `ctx.env.get_current_region()` — Index of currently active region
- `ctx.env.switch_region(idx)` — Switch to region idx (returns True if valid)
- `ctx.env.get_all_regions_spot_available()` — List[bool] of spot availability per region

## Strategy Interface
At each time step, your strategy receives:
- `ctx`: Context with task info and current state
- `last_cluster_type`: What instance was running last tick (pre-preemption: still SPOT if just preempted)
- `has_spot`: Whether spot instance is available this tick IN THE CURRENT REGION

Return: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE

## Available Information
- `ctx.task_duration`: Total work needed (seconds)
- `ctx.deadline`: Deadline time (seconds)
- `ctx.restart_overhead`: Time penalty on restart (seconds)
- `ctx.task_done_time`: List of work done each tick (sum for total progress)
- `ctx.env.elapsed_seconds`: Current time
- `ctx.env.gap_seconds`: Time step size
- `ctx.env.cluster_type`: Current cluster type

## Important Notes
- There is NO migration overhead for switching regions
- has_spot parameter reflects the PRE-switch region; after switching, spot availability
  is checked in the NEW region for the safety override
- Deadline miss = score 0

## Scoring
- Score 0-100 based on cost relative to baselines
- 0 = Cost of running fully on-demand (worst)
- 100 = Cost of running fully on spot without preemptions (best)
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
    Can switch regions before returning using ctx.env.switch_region(idx).

    Args:
        ctx: Strategy context with:
            - ctx.task_duration: Total work needed (seconds)
            - ctx.deadline: Deadline (seconds)
            - ctx.restart_overhead: Restart penalty (seconds)
            - ctx.task_done_time: List of work done per tick
            - ctx.env.elapsed_seconds: Current time
            - ctx.env.gap_seconds: Time step size
            - ctx.env.get_num_regions(): Number of regions
            - ctx.env.get_current_region(): Current region index
            - ctx.env.switch_region(idx): Switch to region idx
            - ctx.env.get_all_regions_spot_available(): Spot availability per region
        last_cluster_type: Previous instance type, pre-preemption (SPOT if just preempted)
        has_spot: Whether spot is available this tick in current region

    Returns:
        ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
    '''
    pass
"""

SEED_PROGRAM = '''import enum
import math

class ClusterType(str, enum.Enum):
    NONE = "NONE"
    SPOT = "SPOT"
    ON_DEMAND = "ON_DEMAND"

def strategy_step(ctx, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
    """Robust stateful multi-region strategy matching ADRS evolutionary_robust_starter.

    Uses linear schedule tracking to decide urgency. When behind schedule,
    aggressively uses ON_DEMAND. When ahead, explores regions for spot.
    """
    env = ctx.env

    work_done = sum(ctx.task_done_time)
    remaining_task_time = ctx.task_duration - work_done
    if remaining_task_time <= 1e-3:
        return ClusterType.NONE

    # Check if behind linear schedule
    t = env.elapsed_seconds
    r_0 = ctx.deadline
    behind_schedule = False
    if r_0 <= t:
        behind_schedule = True
    elif t > 0:
        required_progress = t * (ctx.task_duration / r_0)
        actual_progress = work_done
        behind_schedule = actual_progress < required_progress

    if behind_schedule:
        # URGENT: prioritize getting work done
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.ON_DEMAND
    else:
        # NOT URGENT: prioritize cost savings
        if has_spot:
            return ClusterType.SPOT
        else:
            # Explore: switch to next region round-robin
            current_region = env.get_current_region()
            num_regions = env.get_num_regions()
            next_region = (current_region + 1) % num_regions
            env.switch_region(next_region)
            return ClusterType.NONE
'''

SEED_INSPIRATIONS = []

DIVERSITY_SEED_PROMPT = """
# Can't Be Late: Multi-Region Cloud Scheduling

## Problem
Schedule tasks on spot (cheap, preemptable) vs on-demand (expensive, reliable) instances
across MULTIPLE regions. You can switch between regions to find spot availability.

## Input
- `ctx`: Task context (duration, deadline, restart_overhead, task_done_time, env)
- `last_cluster_type`: Previous instance (NONE/SPOT/ON_DEMAND)
- `has_spot`: Spot availability this tick in current region

## Multi-Region API
- `ctx.env.get_num_regions()` — Number of regions (2-9)
- `ctx.env.get_current_region()` — Current region index
- `ctx.env.switch_region(idx)` — Switch to region idx
- `ctx.env.get_all_regions_spot_available()` — List[bool] per region

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
- switch_region() called with out-of-range index
- Not checking spot availability after switching regions

## Output Format
Keep it SHORT and DIRECT:

**Avoid These Errors:**
- [Error pattern]: [How to fix]

---

{metrics_data}

Provide your lessons (150-200 words max):"""


INPUTS = generate_test_cases()


def score_fn(strategy_step, inputs: List[MultiRegionSimulationConfig]):
    """
    Evaluate multi-region scheduling strategy using ADRS-Leaderboard scoring.

    Scoring matches ADRS evaluator.py evaluate_stage2():
    1. Run all simulations and collect costs
    2. If ANY simulation fails (exception/deadline miss), assign inf cost for that scenario
    3. Group costs by scenario_name, compute per-scenario average
    4. Average of scenario averages -> final_avg
    5. score = clip((od_anchor - final_avg) / (od_anchor - spot_anchor), 0, 1) * 100

    Anchors: od_anchor = 3.06 * 48 = 146.88, spot_anchor = 0.9731 * 48 = 46.7088
    """
    try:
        # Group results by scenario
        scenario_costs = {}  # scenario_name -> list of costs

        for config in inputs:
            cost, met_deadline = simulate(strategy_step, config)

            if not met_deadline:
                return {
                    "score": 0,
                    "num_tests": len(inputs),
                    "avg_cost": float('inf'),
                    "error": f"Deadline missed in scenario {config.scenario_name} trace {config.trace_id}",
                }

            if config.scenario_name not in scenario_costs:
                scenario_costs[config.scenario_name] = []
            scenario_costs[config.scenario_name].append(cost)

        if not scenario_costs:
            return {"error": "No valid test cases"}

        scenario_averages = {}
        for name, costs in scenario_costs.items():
            scenario_averages[name] = sum(costs) / len(costs)

        final_avg = sum(scenario_averages.values()) / len(scenario_averages)

        task_hours = inputs[0].task_duration / 3600
        od_anchor = task_hours * ON_DEMAND_COST_PER_HOUR
        spot_anchor = task_hours * SPOT_COST_PER_HOUR
        denom = od_anchor - spot_anchor

        def _subscore(names):
            avgs = [scenario_averages[n] for n in names if n in scenario_averages]
            if not avgs or denom <= 1e-9:
                return 0.0
            sub_avg = sum(avgs) / len(avgs)
            return max(0.0, min(1.0, (od_anchor - sub_avg) / denom)) * 100

        if denom <= 1e-9:
            score = 0.0
        else:
            score = max(0.0, min(1.0, (od_anchor - final_avg) / denom)) * 100

        return {
            "score": score,
            "num_tests": sum(len(c) for c in scenario_costs.values()),
            "avg_cost": final_avg,
            "od_anchor": od_anchor,
            "spot_anchor": spot_anchor,
            "num_scenarios": len(scenario_costs),
            "few_regions_score": _subscore(["2_zones_same_region", "2_regions_east_west"]),
            "many_regions_score": _subscore(["5_regions_high_diversity", "all_9_regions"]),
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("Multi-Region Test Case Generation")
    print("=" * 50)
    inputs = INPUTS
    print(f"Generated {len(inputs)} test cases")

    # Count per scenario
    from collections import Counter
    counts = Counter(c.scenario_name for c in inputs)
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} traces")

    # Test with seed program
    print("\nTesting seed program...")
    exec_globals = {}
    exec(SEED_PROGRAM, exec_globals)
    seed_fn = exec_globals['strategy_step']

    result = score_fn(seed_fn, inputs)
    print(f"Score: {result.get('score', 'N/A'):.2f}")
    print(f"Avg cost: ${result.get('avg_cost', 'N/A'):.2f}")
    print(f"Details: {result}")
