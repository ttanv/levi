#!/usr/bin/env python3
"""
Comprehensive Diversity Analysis for Can't Be Late Scheduling Runs.

Analyzes:
1. Behavioral diversity (archive coverage, feature spread)
2. Algorithmic diversity (actual code patterns, not just keywords)
3. Homogeneity detection (what do ALL solutions share?)
4. Parameter variation analysis
5. True paradigm detection based on code structure

Usage:
    python analyze_diversity.py                    # Analyze latest run
    python analyze_diversity.py runs/20260128_*   # Analyze specific run
    python analyze_diversity.py --compare         # Compare last N runs
"""

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist


def load_snapshot(run_dir: Path) -> dict:
    """Load snapshot.json from a run directory."""
    snapshot_path = run_dir / "snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"No snapshot.json in {run_dir}")
    with open(snapshot_path) as f:
        return json.load(f)


def find_latest_run(runs_dir: Path) -> Path:
    """Find the most recent run directory."""
    run_dirs = sorted(
        list(runs_dir.glob("*_po")) + list(runs_dir.glob("*_mipe")),
        reverse=True
    )
    if not run_dirs:
        raise FileNotFoundError(f"No runs found in {runs_dir}")
    return run_dirs[0]


# =============================================================================
# BEHAVIORAL DIVERSITY ANALYSIS
# =============================================================================

def analyze_behavioral_diversity(data: dict) -> dict:
    """Analyze behavioral diversity from snapshot data."""
    elites = data["elites"]
    feature_names = data["metadata"]["feature_names"]
    n_centroids = data["metadata"]["n_centroids"]

    # Extract behavior vectors
    behaviors = np.array([
        [e["behavior"][name] for name in feature_names]
        for e in elites
    ])

    # Archive coverage
    unique_cells = len(set(e["cell_index"] for e in elites))
    coverage = unique_cells / n_centroids

    # Unique behavioral profiles at different precisions
    unique_01 = len(set(tuple(np.round(b, 2)) for b in behaviors))
    unique_1 = len(set(tuple(np.round(b, 1)) for b in behaviors))

    # Pairwise distances
    if len(behaviors) > 1:
        dists = pdist(behaviors, metric="euclidean")
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        min_dist = np.min(dists)
        max_dist = np.max(dists)
    else:
        mean_dist = std_dist = min_dist = max_dist = 0.0

    # Feature-wise statistics
    feature_stats = {}
    for i, name in enumerate(feature_names):
        vals = behaviors[:, i]
        feature_stats[name] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "spread": float(vals.max() - vals.min()),
        }

    # Score distribution
    scores = [e["primary_score"] for e in elites]

    return {
        "n_elites": len(elites),
        "n_centroids": n_centroids,
        "coverage": coverage,
        "unique_profiles_01": unique_01,
        "unique_profiles_1": unique_1,
        "pairwise_distance": {
            "mean": mean_dist,
            "std": std_dist,
            "min": min_dist,
            "max": max_dist,
        },
        "feature_stats": feature_stats,
        "score_stats": {
            "min": min(scores),
            "max": max(scores),
            "mean": np.mean(scores),
            "std": np.std(scores),
        },
    }


# =============================================================================
# CORE ALGORITHM DETECTION (Structure-based, not keyword-based)
# =============================================================================

def extract_decision_structure(code: str) -> dict:
    """Extract the actual decision-making structure from code."""
    structure = {
        # Core patterns (what the code ACTUALLY does)
        "has_slack_calculation": bool(re.search(r"slack\s*=\s*[^=]", code)),
        "has_buffer_calculation": bool(re.search(r"buffer\s*=\s*[^=]", code)),
        "has_risk_calculation": bool(re.search(r"risk\s*=\s*[^=]", code)),
        "checks_has_spot": "has_spot" in code,
        "checks_last_cluster": "last_cluster" in code,
        "uses_restart_overhead": "restart_overhead" in code,
        "uses_time_remaining": bool(re.search(r"time_remaining|remaining.*time|deadline\s*-\s*elapsed", code)),
        "uses_work_remaining": bool(re.search(r"work_remaining|work_left|task_duration\s*-", code)),

        # Decision thresholds
        "n_numeric_comparisons": len(re.findall(r"(?:slack|buffer|time|work)\s*[<>=]+\s*[\d\.]+", code)),
        "n_if_statements": code.count("if "),
        "n_return_statements": len(re.findall(r"return\s+ClusterType", code)),

        # State machine patterns
        "state_dependent_logic": bool(re.search(r"if\s+last_cluster.*:.*\n.*(?:if|return)", code, re.DOTALL)),
        "hysteresis_pattern": bool(re.search(r"last_cluster.*==.*SPOT.*:.*SPOT|stay|continue", code, re.IGNORECASE)),

        # Mathematical patterns (actual computations, not just mentions)
        "computes_ratio": bool(re.search(r"(?:work|time|slack).*[/].*(?:work|time|slack)", code)),
        "uses_multiplication_factor": bool(re.search(r"(?:work|time|slack|buffer).*\*\s*[\d\.]+", code)),
    }
    return structure


def detect_true_paradigm(code: str) -> tuple[str, float]:
    """
    Detect the TRUE algorithmic paradigm based on code structure.
    Returns (paradigm_name, confidence).

    Looks for actual implementation patterns, not just keyword mentions.
    """
    code_lower = code.lower()

    # Dynamic Programming: Must have actual DP array/memoization
    if re.search(r"dp\s*\[|memo\s*\[|@cache|@lru_cache", code):
        if re.search(r"dp\s*\[.*\]\s*=|memo\s*\[.*\]\s*=", code):
            return ("dynamic_programming", 0.9)

    # Monte Carlo: Must have actual simulation loop
    if re.search(r"for\s+\w+\s+in\s+range.*:.*random|sample|simulate", code, re.DOTALL):
        if "random" in code and re.search(r"for.*range.*\d{2,}", code):  # Multiple iterations
            return ("monte_carlo_simulation", 0.8)

    # Markov/MDP: Must have transition probabilities or value iteration
    if re.search(r"transition.*prob|p_preempt|probability.*\d\.\d", code_lower):
        if re.search(r"expect|value.*=.*prob", code_lower):
            return ("stochastic_mdp", 0.7)

    # Phase-based: Must have distinct phases with different logic
    phase_markers = len(re.findall(r"phase|stage|early|middle|late", code_lower))
    if phase_markers >= 2:
        if re.search(r"if.*(?:early|phase.*1|elapsed.*<).*:.*\n.*(?:elif|if).*(?:late|phase.*2|elapsed.*>)", code, re.DOTALL | re.IGNORECASE):
            return ("phase_based", 0.75)

    # History-adaptive: Must actually track and use history
    if re.search(r"history|past_|prev_|track", code_lower):
        if re.search(r"history\s*\[|\.append|sum\(.*history|len\(.*history", code):
            return ("history_adaptive", 0.7)

    # Lookahead: Must compute future states
    if re.search(r"lookahead|future|next_\d|predict", code_lower):
        if re.search(r"for.*range.*:.*if.*future|next_tick", code):
            return ("lookahead", 0.65)

    # Now check for the DOMINANT pattern: Greedy Slack-Buffer
    has_slack = bool(re.search(r"slack\s*=", code))
    has_buffer = bool(re.search(r"buffer\s*=", code))
    has_threshold = bool(re.search(r"if\s+(?:slack|buffer).*[<>]", code))
    has_spot_check = "has_spot" in code
    has_cluster_state = "last_cluster" in code

    # This is the common pattern
    if has_slack and has_threshold and has_spot_check:
        if has_buffer:
            return ("greedy_slack_buffer", 0.95)
        return ("greedy_threshold", 0.9)

    if has_threshold and has_spot_check and has_cluster_state:
        return ("greedy_state_machine", 0.85)

    # Fallback
    if has_spot_check:
        return ("greedy_heuristic", 0.7)

    return ("unknown", 0.5)


def analyze_homogeneity(data: dict) -> dict:
    """
    Analyze what ALL or MOST solutions share.
    High homogeneity = low algorithmic diversity.
    """
    elites = data["elites"]
    n = len(elites)

    # Check for patterns present in ALL/MOST solutions
    patterns = {
        "slack_calculation": 0,
        "buffer_calculation": 0,
        "has_spot_check": 0,
        "last_cluster_check": 0,
        "restart_overhead_handling": 0,
        "time_remaining_check": 0,
        "work_remaining_check": 0,
        "state_machine_pattern": 0,
        "threshold_comparison": 0,
        "hysteresis_logic": 0,
    }

    for e in elites:
        code = e["code"]
        if re.search(r"slack\s*=", code):
            patterns["slack_calculation"] += 1
        if re.search(r"buffer\s*=", code):
            patterns["buffer_calculation"] += 1
        if "has_spot" in code:
            patterns["has_spot_check"] += 1
        if "last_cluster" in code:
            patterns["last_cluster_check"] += 1
        if "restart_overhead" in code:
            patterns["restart_overhead_handling"] += 1
        if re.search(r"time_remaining|remaining.*time|deadline\s*-", code):
            patterns["time_remaining_check"] += 1
        if re.search(r"work_remaining|work_left|task_duration\s*-", code):
            patterns["work_remaining_check"] += 1
        if re.search(r"if\s+last_cluster.*:.*\n.*if", code, re.DOTALL):
            patterns["state_machine_pattern"] += 1
        if re.search(r"if\s+(?:slack|buffer).*[<>]\s*[\d\.]", code):
            patterns["threshold_comparison"] += 1
        if re.search(r"last_cluster.*SPOT.*SPOT|stay|continue.*spot", code, re.IGNORECASE):
            patterns["hysteresis_logic"] += 1

    # Calculate percentages
    pattern_pcts = {k: v / n * 100 for k, v in patterns.items()}

    # Patterns present in >80% are "universal"
    universal = [k for k, v in pattern_pcts.items() if v >= 80]
    common = [k for k, v in pattern_pcts.items() if 50 <= v < 80]
    rare = [k for k, v in pattern_pcts.items() if v < 50]

    # Homogeneity score: high if many universal patterns
    homogeneity_score = len(universal) / len(patterns) * 100

    return {
        "pattern_percentages": pattern_pcts,
        "universal_patterns": universal,
        "common_patterns": common,
        "rare_patterns": rare,
        "homogeneity_score": homogeneity_score,
    }


# =============================================================================
# ALGORITHMIC DIVERSITY ANALYSIS
# =============================================================================

def analyze_algorithmic_diversity(data: dict) -> dict:
    """Analyze TRUE algorithmic diversity from snapshot data."""
    elites = data["elites"]

    # Detect true paradigms
    paradigm_results = [detect_true_paradigm(e["code"]) for e in elites]
    paradigms = [p[0] for p in paradigm_results]
    confidences = [p[1] for p in paradigm_results]

    paradigm_counts = Counter(paradigms)

    # Paradigm-score correlation
    paradigm_scores = defaultdict(list)
    for e, p in zip(elites, paradigms):
        paradigm_scores[p].append(e["primary_score"])

    paradigm_stats = {
        p: {
            "count": paradigm_counts[p],
            "pct": paradigm_counts[p] / len(elites) * 100,
            "best_score": max(scores),
            "avg_score": np.mean(scores),
        }
        for p, scores in paradigm_scores.items()
    }

    # Decision structure analysis
    structures = [extract_decision_structure(e["code"]) for e in elites]

    # Code metrics
    code_lengths = [len(e["code"]) for e in elites]
    line_counts = [e["code"].count("\n") + 1 for e in elites]
    if_counts = [e["code"].count("if ") for e in elites]
    return_counts = [len(re.findall(r"return\s+ClusterType", e["code"])) for e in elites]

    # Parameter extraction
    all_params = []
    for e in elites:
        code = e["code"]
        params = {}

        # Risk/buffer multipliers
        match = re.search(r"work_(?:left|remaining)\s*\*\s*([\d\.]+)", code)
        if match:
            params["risk_factor"] = float(match.group(1))

        # Tick multipliers
        match = re.search(r"(\d+)\s*\*\s*(?:gap|tick)", code)
        if match:
            params["tick_buffer"] = int(match.group(1))

        # Threshold values
        thresholds = re.findall(r"slack\s*[<>=]+\s*([\d\.]+)", code)
        if thresholds:
            params["slack_thresholds"] = [float(t) for t in thresholds[:3]]

        all_params.append(params)

    # Aggregate parameter stats
    risk_factors = [p["risk_factor"] for p in all_params if "risk_factor" in p]
    tick_buffers = [p["tick_buffer"] for p in all_params if "tick_buffer" in p]

    param_diversity = {
        "risk_factor": {
            "unique_values": sorted(set(risk_factors)) if risk_factors else [],
            "n_unique": len(set(risk_factors)) if risk_factors else 0,
        },
        "tick_buffer": {
            "unique_values": sorted(set(tick_buffers)) if tick_buffers else [],
            "n_unique": len(set(tick_buffers)) if tick_buffers else 0,
        },
    }

    # Calculate TRUE diversity score
    # Low diversity if: dominated by one paradigm AND high homogeneity
    dominant_pct = max(paradigm_counts.values()) / len(elites) * 100
    n_paradigms = len([p for p, c in paradigm_counts.items() if c >= 2])  # Paradigms with 2+ solutions

    # Algorithmic diversity score (0-100)
    # Penalize: single dominant paradigm, few unique paradigms
    algo_diversity_score = 100
    if dominant_pct > 50:
        algo_diversity_score -= (dominant_pct - 50)  # Penalty for dominance
    if n_paradigms < 3:
        algo_diversity_score -= (3 - n_paradigms) * 15  # Penalty for few paradigms
    algo_diversity_score = max(0, algo_diversity_score)

    return {
        "paradigms": paradigm_stats,
        "dominant_paradigm": paradigm_counts.most_common(1)[0] if paradigm_counts else None,
        "n_unique_paradigms": len(paradigm_counts),
        "n_significant_paradigms": n_paradigms,
        "dominant_pct": dominant_pct,
        "algo_diversity_score": algo_diversity_score,
        "avg_confidence": np.mean(confidences),
        "code_stats": {
            "length": {"min": min(code_lengths), "max": max(code_lengths), "mean": np.mean(code_lengths)},
            "lines": {"min": min(line_counts), "max": max(line_counts), "mean": np.mean(line_counts)},
            "if_stmts": {"min": min(if_counts), "max": max(if_counts), "mean": np.mean(if_counts)},
            "return_stmts": {"min": min(return_counts), "max": max(return_counts), "mean": np.mean(return_counts)},
        },
        "param_diversity": param_diversity,
    }


# =============================================================================
# MISSING PARADIGMS CHECK
# =============================================================================

TRULY_DIFFERENT_PARADIGMS = {
    "dynamic_programming": "Optimal substructure with memoization/tabulation",
    "monte_carlo_simulation": "Simulate future scenarios to pick best action",
    "stochastic_mdp": "Model as Markov Decision Process with probabilities",
    "phase_based": "Different strategies for early/middle/late execution",
    "history_adaptive": "Learn from observed spot availability patterns",
    "lookahead": "Compute expected outcomes N steps ahead",
    "constraint_satisfaction": "Model as constraint satisfaction problem",
}

GREEDY_VARIANTS = {
    "greedy_slack_buffer": "Slack calculation with safety buffer",
    "greedy_threshold": "Simple threshold-based decisions",
    "greedy_state_machine": "State-dependent greedy decisions",
    "greedy_heuristic": "Basic greedy heuristic",
}


def check_paradigm_coverage(data: dict) -> dict:
    """Check which truly different paradigms are present."""
    elites = data["elites"]

    # Detect paradigms for all solutions
    paradigms = [detect_true_paradigm(e["code"])[0] for e in elites]
    paradigm_set = set(paradigms)

    # Check truly different paradigms
    truly_different_present = []
    truly_different_missing = []

    for p, desc in TRULY_DIFFERENT_PARADIGMS.items():
        if p in paradigm_set:
            truly_different_present.append(p)
        else:
            truly_different_missing.append(p)

    # Check greedy variants
    greedy_variants_present = [p for p in GREEDY_VARIANTS if p in paradigm_set]

    # Is population mostly greedy?
    greedy_count = sum(1 for p in paradigms if p in GREEDY_VARIANTS)
    greedy_pct = greedy_count / len(paradigms) * 100

    return {
        "truly_different_present": truly_different_present,
        "truly_different_missing": truly_different_missing,
        "truly_different_coverage": len(truly_different_present) / len(TRULY_DIFFERENT_PARADIGMS) * 100,
        "greedy_variants_present": greedy_variants_present,
        "greedy_dominance_pct": greedy_pct,
        "is_greedy_dominated": greedy_pct > 70,
    }


# =============================================================================
# COMPREHENSIVE DIVERSITY SCORE
# =============================================================================

def compute_overall_diversity(behavioral: dict, algorithmic: dict, homogeneity: dict, paradigm_coverage: dict) -> dict:
    """Compute an overall diversity assessment."""

    # Behavioral score (0-100)
    behav_score = behavioral["coverage"] * 100

    # Algorithmic score (already 0-100)
    algo_score = algorithmic["algo_diversity_score"]

    # Homogeneity penalty (high homogeneity = low diversity)
    homogeneity_penalty = homogeneity["homogeneity_score"]

    # Paradigm coverage bonus
    paradigm_bonus = paradigm_coverage["truly_different_coverage"]

    # Greedy dominance penalty
    greedy_penalty = max(0, paradigm_coverage["greedy_dominance_pct"] - 50)

    # Combined score
    overall = (
        behav_score * 0.2 +  # Behavioral matters less
        algo_score * 0.3 +
        paradigm_bonus * 0.3 +
        (100 - homogeneity_penalty) * 0.1 +
        (100 - greedy_penalty) * 0.1
    )

    # Ratings
    if overall >= 70 and paradigm_coverage["truly_different_coverage"] >= 50:
        rating = "HIGH"
    elif overall >= 50 or paradigm_coverage["truly_different_coverage"] >= 30:
        rating = "MEDIUM"
    else:
        rating = "LOW"

    return {
        "overall_score": overall,
        "rating": rating,
        "components": {
            "behavioral": behav_score,
            "algorithmic": algo_score,
            "paradigm_coverage": paradigm_bonus,
            "homogeneity_penalty": homogeneity_penalty,
            "greedy_penalty": greedy_penalty,
        },
    }


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(run_dir: Path, data: dict) -> str:
    """Generate a comprehensive diversity report."""
    behavioral = analyze_behavioral_diversity(data)
    algorithmic = analyze_algorithmic_diversity(data)
    homogeneity = analyze_homogeneity(data)
    paradigm_cov = check_paradigm_coverage(data)
    overall = compute_overall_diversity(behavioral, algorithmic, homogeneity, paradigm_cov)

    lines = []

    # Header
    lines.append("=" * 80)
    lines.append(f"DIVERSITY ANALYSIS REPORT: {run_dir.name}")
    lines.append("=" * 80)
    lines.append("")

    # Overall Summary
    lines.append("┌" + "─" * 78 + "┐")
    lines.append("│" + "OVERALL DIVERSITY ASSESSMENT".center(78) + "│")
    lines.append("├" + "─" * 78 + "┤")

    rating_color = {"HIGH": "✓", "MEDIUM": "~", "LOW": "✗"}[overall["rating"]]
    lines.append(f"│  Overall Diversity:      {rating_color} {overall['rating']:6s} (Score: {overall['overall_score']:.1f}/100)".ljust(78) + " │")
    lines.append(f"│  ".ljust(78) + " │")
    lines.append(f"│  Behavioral Coverage:    {behavioral['coverage']*100:.0f}% ({behavioral['unique_profiles_01']}/{behavioral['n_centroids']} unique)".ljust(78) + " │")
    lines.append(f"│  Algorithmic Diversity:  {algorithmic['algo_diversity_score']:.0f}/100 ({algorithmic['n_significant_paradigms']} significant paradigms)".ljust(78) + " │")
    lines.append(f"│  Paradigm Coverage:      {paradigm_cov['truly_different_coverage']:.0f}% ({len(paradigm_cov['truly_different_present'])}/{len(TRULY_DIFFERENT_PARADIGMS)} different approaches)".ljust(78) + " │")
    lines.append(f"│  Greedy Dominance:       {paradigm_cov['greedy_dominance_pct']:.0f}% {'⚠ HIGH' if paradigm_cov['is_greedy_dominated'] else ''}".ljust(78) + " │")
    lines.append(f"│  ".ljust(78) + " │")
    lines.append(f"│  Best Score:             {behavioral['score_stats']['max']:.2f}".ljust(78) + " │")
    lines.append(f"│  Mean Score:             {behavioral['score_stats']['mean']:.2f} ± {behavioral['score_stats']['std']:.2f}".ljust(78) + " │")
    lines.append("└" + "─" * 78 + "┘")
    lines.append("")

    # Homogeneity Analysis (NEW - what do ALL solutions share?)
    lines.append("─" * 80)
    lines.append("HOMOGENEITY ANALYSIS (What ALL/MOST solutions share)")
    lines.append("─" * 80)
    lines.append("")
    lines.append(f"  Homogeneity Score: {homogeneity['homogeneity_score']:.0f}% (lower is more diverse)")
    lines.append("")

    if homogeneity["universal_patterns"]:
        lines.append("  Universal Patterns (>80% of solutions):")
        for p in homogeneity["universal_patterns"]:
            pct = homogeneity["pattern_percentages"][p]
            lines.append(f"    ⚠ {p:35s}: {pct:.0f}%")
        lines.append("")

    if homogeneity["common_patterns"]:
        lines.append("  Common Patterns (50-80%):")
        for p in homogeneity["common_patterns"]:
            pct = homogeneity["pattern_percentages"][p]
            lines.append(f"    ~ {p:35s}: {pct:.0f}%")
        lines.append("")

    if homogeneity["rare_patterns"]:
        lines.append("  Rare Patterns (<50%):")
        for p in homogeneity["rare_patterns"]:
            pct = homogeneity["pattern_percentages"][p]
            lines.append(f"    ✓ {p:35s}: {pct:.0f}%")
        lines.append("")

    # Algorithmic Paradigms
    lines.append("─" * 80)
    lines.append("ALGORITHMIC PARADIGMS (Structure-based detection)")
    lines.append("─" * 80)
    lines.append("")

    # Separate greedy variants from truly different
    lines.append("  Greedy Variants (similar core algorithm):")
    for paradigm, stats in sorted(algorithmic["paradigms"].items(), key=lambda x: -x[1]["count"]):
        if paradigm in GREEDY_VARIANTS:
            lines.append(f"    {paradigm:30s}: {stats['count']:3d} ({stats['pct']:5.1f}%) best={stats['best_score']:.2f}")
    lines.append("")

    truly_diff_found = [p for p in algorithmic["paradigms"] if p in TRULY_DIFFERENT_PARADIGMS]
    if truly_diff_found:
        lines.append("  Truly Different Approaches:")
        for paradigm in truly_diff_found:
            stats = algorithmic["paradigms"][paradigm]
            lines.append(f"    ✓ {paradigm:28s}: {stats['count']:3d} ({stats['pct']:5.1f}%) best={stats['best_score']:.2f}")
        lines.append("")

    if paradigm_cov["truly_different_missing"]:
        lines.append("  Missing Paradigms (NOT found in population):")
        for p in paradigm_cov["truly_different_missing"]:
            desc = TRULY_DIFFERENT_PARADIGMS[p]
            lines.append(f"    ✗ {p:28s}: {desc}")
        lines.append("")

    # Parameter Variation
    lines.append("─" * 80)
    lines.append("PARAMETER VARIATION")
    lines.append("─" * 80)
    lines.append("")

    pd = algorithmic["param_diversity"]
    if pd["risk_factor"]["unique_values"]:
        lines.append(f"  Risk Factor (work_left * X):")
        lines.append(f"    Values: {pd['risk_factor']['unique_values']}")
        lines.append(f"    Unique: {pd['risk_factor']['n_unique']}")
    else:
        lines.append(f"  Risk Factor: Not detected")
    lines.append("")

    if pd["tick_buffer"]["unique_values"]:
        lines.append(f"  Tick Buffer (N * gap):")
        lines.append(f"    Values: {pd['tick_buffer']['unique_values']}")
        lines.append(f"    Unique: {pd['tick_buffer']['n_unique']}")
    else:
        lines.append(f"  Tick Buffer: Not detected")
    lines.append("")

    # Code Statistics
    lines.append("─" * 80)
    lines.append("CODE STATISTICS")
    lines.append("─" * 80)
    lines.append("")
    cs = algorithmic["code_stats"]
    lines.append(f"  Lines:        {cs['lines']['min']:.0f} - {cs['lines']['max']:.0f} (avg {cs['lines']['mean']:.0f})")
    lines.append(f"  Characters:   {cs['length']['min']:.0f} - {cs['length']['max']:.0f} (avg {cs['length']['mean']:.0f})")
    lines.append(f"  If Stmts:     {cs['if_stmts']['min']:.0f} - {cs['if_stmts']['max']:.0f} (avg {cs['if_stmts']['mean']:.1f})")
    lines.append(f"  Return Stmts: {cs['return_stmts']['min']:.0f} - {cs['return_stmts']['max']:.0f} (avg {cs['return_stmts']['mean']:.1f})")
    lines.append("")

    # Behavioral Diversity (brief)
    lines.append("─" * 80)
    lines.append("BEHAVIORAL DIVERSITY (Archive Coverage)")
    lines.append("─" * 80)
    lines.append("")
    lines.append(f"  Coverage:     {behavioral['coverage']*100:.0f}% ({behavioral['unique_profiles_01']}/{behavioral['n_centroids']})")
    lines.append(f"  Mean Dist:    {behavioral['pairwise_distance']['mean']:.3f}")
    lines.append("")
    lines.append("  Feature Spread:")
    for name, stats in behavioral["feature_stats"].items():
        lines.append(f"    {name:25s}: {stats['min']:.2f} - {stats['max']:.2f} (spread: {stats['spread']:.2f})")
    lines.append("")

    # Diagnosis & Recommendations
    lines.append("─" * 80)
    lines.append("DIAGNOSIS & RECOMMENDATIONS")
    lines.append("─" * 80)
    lines.append("")

    issues = []

    if paradigm_cov["is_greedy_dominated"]:
        issues.append(f"• CRITICAL: {paradigm_cov['greedy_dominance_pct']:.0f}% of solutions are greedy variants.")
        issues.append("  All solutions use the same core algorithm with minor variations.")
        issues.append("  → Need fundamentally different approaches (DP, MDP, Monte Carlo)")

    if homogeneity["homogeneity_score"] > 60:
        issues.append(f"• HIGH HOMOGENEITY: {len(homogeneity['universal_patterns'])} patterns shared by >80% of solutions.")
        issues.append(f"  Shared: {', '.join(homogeneity['universal_patterns'][:4])}")
        issues.append("  → Population has converged to one approach with parameter tweaks")

    if paradigm_cov["truly_different_coverage"] < 30:
        issues.append(f"• LOW PARADIGM COVERAGE: Only {len(paradigm_cov['truly_different_present'])}/{len(TRULY_DIFFERENT_PARADIGMS)} different paradigms found.")
        issues.append(f"  Missing: {', '.join(paradigm_cov['truly_different_missing'][:3])}")
        issues.append("  → Seed population with diverse algorithmic baselines")

    if algorithmic["param_diversity"]["risk_factor"]["n_unique"] <= 2:
        issues.append("• LOW PARAMETER DIVERSITY: Very few unique parameter values.")
        issues.append("  → Solutions are nearly identical, differing only slightly")

    if not issues:
        issues.append("• Population shows reasonable diversity.")
        issues.append("  Consider exploring more truly different paradigms for potential gains.")

    for issue in issues:
        lines.append(f"  {issue}")
    lines.append("")

    return "\n".join(lines)


def generate_comparison(run_dirs: list[Path]) -> str:
    """Generate comparison of multiple runs."""
    lines = []
    lines.append("=" * 100)
    lines.append("DIVERSITY COMPARISON OF RECENT RUNS")
    lines.append("=" * 100)
    lines.append("")

    # Header
    lines.append(f"{'Run':<25} {'Score':>8} {'Behav%':>8} {'Algo':>6} {'Paradigms':>10} {'Greedy%':>9} {'Homogen%':>10} {'Rating':>8}")
    lines.append("-" * 100)

    for run_dir in run_dirs:
        try:
            data = load_snapshot(run_dir)
            behavioral = analyze_behavioral_diversity(data)
            algorithmic = analyze_algorithmic_diversity(data)
            homogeneity = analyze_homogeneity(data)
            paradigm_cov = check_paradigm_coverage(data)
            overall = compute_overall_diversity(behavioral, algorithmic, homogeneity, paradigm_cov)

            lines.append(
                f"{run_dir.name:<25} "
                f"{behavioral['score_stats']['max']:>8.2f} "
                f"{behavioral['coverage']*100:>7.0f}% "
                f"{algorithmic['algo_diversity_score']:>6.0f} "
                f"{algorithmic['n_significant_paradigms']:>10} "
                f"{paradigm_cov['greedy_dominance_pct']:>8.0f}% "
                f"{homogeneity['homogeneity_score']:>9.0f}% "
                f"{overall['rating']:>8}"
            )
        except Exception as e:
            lines.append(f"{run_dir.name:<25} Error: {str(e)[:50]}")

    lines.append("")
    lines.append("Legend:")
    lines.append("  Score     = Best score in population")
    lines.append("  Behav%    = Behavioral archive coverage")
    lines.append("  Algo      = Algorithmic diversity score (0-100)")
    lines.append("  Paradigms = Number of significantly different paradigms")
    lines.append("  Greedy%   = Percentage of solutions using greedy variants")
    lines.append("  Homogen%  = Homogeneity score (lower = more diverse)")
    lines.append("  Rating    = Overall diversity rating")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze diversity in Can't Be Late runs")
    parser.add_argument("run_dir", nargs="?", help="Run directory to analyze (default: latest)")
    parser.add_argument("--compare", action="store_true", help="Compare last N runs")
    parser.add_argument("-n", type=int, default=5, help="Number of runs to compare (with --compare)")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of report")
    parser.add_argument("--output", "-o", type=str, help="Save report to file")
    args = parser.parse_args()

    # Find runs directory
    script_dir = Path(__file__).parent
    runs_dir = script_dir.parent.parent / "runs"

    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    if args.compare:
        run_dirs = sorted(
            list(runs_dir.glob("*_po")) + list(runs_dir.glob("*_mipe")),
            reverse=True
        )[:args.n]

        if not run_dirs:
            print("No runs found to compare", file=sys.stderr)
            sys.exit(1)

        output = generate_comparison(run_dirs)
    else:
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            run_dir = find_latest_run(runs_dir)

        try:
            data = load_snapshot(run_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            result = {
                "run": run_dir.name,
                "behavioral": analyze_behavioral_diversity(data),
                "algorithmic": analyze_algorithmic_diversity(data),
                "homogeneity": analyze_homogeneity(data),
                "paradigm_coverage": check_paradigm_coverage(data),
            }
            result["overall"] = compute_overall_diversity(
                result["behavioral"], result["algorithmic"],
                result["homogeneity"], result["paradigm_coverage"]
            )
            output = json.dumps(result, indent=2, default=str)
        else:
            output = generate_report(run_dir, data)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report saved to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
