import enum

class ClusterType(str, enum.Enum):
    NONE = "NONE"
    SPOT = "SPOT"
    ON_DEMAND = "ON_DEMAND"

def strategy_step(ctx, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
    """
    Multi-Region Cloud Instance Scheduling Strategy v2

    Optimized for:
    - Deadline compliance (0 score = failure)
    - Cost minimization via spot instance exploitation
    - Efficient cross-region switching with validation
    - Avoiding over-reliance on correlated regions

    Logic Flow:
    1. Deadline Safety Check (Critical)
    2. Current Region Spot Available? → Use SPOT
    3. Else: Search ALL regions for SPOT availability → Switch to first available
    4. If no spot anywhere: use ON_DEMAND if behind schedule, else wait (NONE) if safe
    """

    env = ctx.env
    now = env.elapsed_seconds
    deadline = ctx.deadline
    total_work = ctx.task_duration
    done_work = sum(ctx.task_done_time)
    remaining_work = total_work - done_work
    time_left = deadline - now

    # Task already finished
    if remaining_work <= 0:
        return ClusterType.NONE

    # --- 1. DEADLINE SAFETY: Point of No Return (PNR) ---
    # Required time to finish, including restart overhead
    required_time = remaining_work + ctx.restart_overhead
    # Add 5% buffer to account for jitter and correlated failures
    safety_threshold = required_time * 1.05

    if time_left <= safety_threshold:
        # Cannot afford risk. Must complete now.
        # Use SPOT if available (even if just preempted), else ON_DEMAND
        return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

    # --- 2. Check Current Region Spot Availability ---
    if has_spot:
        return ClusterType.SPOT

    # --- 3. Multi-Region Spot Arbitrage: Find Any Available Spot ---
    all_spots = env.get_all_regions_spot_available()
    num_regions = env.get_num_regions()
    current_region = env.get_current_region()

    # Try to switch to a region with spot availability
    for idx in range(num_regions):
        if all_spots[idx]:
            # Switch to this region and use SPOT immediately
            if env.switch_region(idx):
                return ClusterType.SPOT

    # --- 4. No Spot Available Anywhere: Decide Between ON_DEMAND or NONE ---
    # Calculate ideal progress rate and current deviation
    ideal_rate = total_work / deadline
    expected_progress = ideal_rate * now
    progress_deviation = done_work - expected_progress  # Positive = ahead, negative = behind

    # If behind schedule, cannot afford to wait
    if progress_deviation < 0:
        return ClusterType.ON_DEMAND

    # If ahead, we can afford to wait — but only if we have enough slack
    slack = time_left - remaining_work
    # Minimum slack threshold: 1 hour or 5% of deadline
    min_slack = max(3600.0, deadline * 0.05)

    if slack > min_slack:
        # Safe to wait. Use NONE and rotate regions occasionally to monitor changes.
        # Rotate every 10 ticks to avoid API spam and detect dynamic availability
        tick_index = int(now // env.gap_seconds)
        if tick_index % 10 == 0:
            next_region = (current_region + 1) % num_regions
            env.switch_region(next_region)
        return ClusterType.NONE

    # Otherwise: not far enough ahead to risk waiting
    return ClusterType.ON_DEMAND