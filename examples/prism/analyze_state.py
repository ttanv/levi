import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the state file
with open('state.json', 'r') as f:
    state = json.load(f)

# Extract metadata
metadata = state['metadata']
print("=== METADATA ===")
print(f"Archive size: {metadata['archive_size']}")
print(f"Best score: {metadata['best_score']}")
print(f"Feature names (behavioral dimensions): {metadata['feature_names']}")
print(f"Learned bounds: {json.dumps(metadata['learned_bounds'], indent=2)}")
print()

# Extract all elites
elites = state['elites']
print(f"=== ELITES ({len(elites)} total) ===")

# Collect scores and behaviors
scores = []
behaviors = []
cells = []
created_times = []

for elite in elites:
    scores.append(elite['primary_score'])
    behaviors.append(elite['behavior'])
    cells.append(elite['cell_index'])
    created_times.append(elite.get('created_at', ''))

# Sort by score
sorted_indices = np.argsort(scores)[::-1]  # Descending (higher is better)

print("\n=== SCORE DISTRIBUTION ===")
print(f"Min score: {min(scores):.4f}")
print(f"Max score: {max(scores):.4f}")
print(f"Mean score: {np.mean(scores):.4f}")
print(f"Std score: {np.std(scores):.4f}")

# Find solutions around 87.4
print("\n=== SOLUTIONS AROUND 87.4 ===")
around_87 = [(i, scores[i]) for i in range(len(scores)) if 86.5 <= scores[i] <= 88.5]
print(f"Found {len(around_87)} solutions between 86.5 and 88.5")
for i, s in sorted(around_87, key=lambda x: x[1]):
    print(f"  Index {i}: score={s:.4f}, behavior={behaviors[i]}")

# Top 5 solutions
print("\n=== TOP 5 SOLUTIONS ===")
for rank, idx in enumerate(sorted_indices[:5]):
    print(f"Rank {rank+1}: score={scores[idx]:.4f}, cell={cells[idx]}")
    print(f"  Behavior: {behaviors[idx]}")
    print()

# Behavioral diversity analysis
print("\n=== BEHAVIORAL DIVERSITY ===")
feature_names = metadata['feature_names']
for feat in feature_names:
    values = [b[feat] for b in behaviors]
    print(f"{feat}:")
    print(f"  Min: {min(values):.4f}, Max: {max(values):.4f}")
    print(f"  Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")
    print(f"  Range covered: {max(values) - min(values):.4f}")

# Analyze cell coverage
print("\n=== CELL COVERAGE ===")
unique_cells = set(cells)
print(f"Unique cells occupied: {len(unique_cells)} out of {metadata['n_centroids']} centroids")
print(f"Cell indices: {sorted(unique_cells)}")

# Algorithmic diversity - look at code structure
print("\n=== ALGORITHMIC DIVERSITY ===")
algo_types = defaultdict(list)

for i, elite in enumerate(elites):
    code = elite['code']
    features = []
    if 'simulated_annealing' in code.lower():
        features.append('simulated_annealing')
    if 'local_search' in code.lower():
        features.append('local_search')
    if 'greedy' in code.lower():
        features.append('greedy')
    if 'random' in code.lower():
        features.append('random_restart')
    if 'genetic' in code.lower() or 'crossover' in code.lower():
        features.append('genetic')
    if 'tabu' in code.lower():
        features.append('tabu_search')
    if 'beam' in code.lower():
        features.append('beam_search')
    if 'dynamic' in code.lower() or 'dp_' in code.lower():
        features.append('dynamic_programming')
    if 'gradient' in code.lower():
        features.append('gradient')
    if 'branch' in code.lower() and 'bound' in code.lower():
        features.append('branch_and_bound')

    # Count iterations
    import re
    iter_matches = re.findall(r'iterations?\s*=\s*(\d+)', code, re.IGNORECASE)
    if iter_matches:
        features.append(f'iterations:{max(int(x) for x in iter_matches)}')

    restart_matches = re.findall(r'(?:restart|attempt).*?(?:range\s*\(\s*(\d+)|for.*?in.*?range\s*\(\s*(\d+))', code, re.IGNORECASE)

    algo_types[tuple(sorted(set(features)))].append((i, scores[i]))

print("Algorithm combinations found:")
for algo_combo, instances in sorted(algo_types.items(), key=lambda x: -len(x[1])):
    avg_score = np.mean([s for _, s in instances])
    print(f"  {algo_combo}: {len(instances)} solutions, avg score: {avg_score:.4f}")

# Create budget vs score plot
print("\n=== GENERATING PLOTS ===")

# Sort by creation time to show progression
# Parse timestamps and compute budget percentage
from datetime import datetime

# Get creation times and sort
time_score_pairs = []
for i, elite in enumerate(elites):
    created = elite.get('created_at', '')
    if created:
        try:
            dt = datetime.fromisoformat(created)
            time_score_pairs.append((dt, scores[i], i))
        except:
            pass

if time_score_pairs:
    time_score_pairs.sort(key=lambda x: x[0])

    # Compute budget as percentage of total evaluations
    min_time = time_score_pairs[0][0]
    max_time = time_score_pairs[-1][0]
    total_duration = (max_time - min_time).total_seconds()

    budget_pct = []
    score_progress = []
    best_so_far = 0

    for dt, score, idx in time_score_pairs:
        elapsed = (dt - min_time).total_seconds()
        pct = (elapsed / total_duration * 100) if total_duration > 0 else 0
        budget_pct.append(pct)
        best_so_far = max(best_so_far, score)
        score_progress.append(best_so_far)

    # Calculate score increase from baseline
    baseline = min(scores)
    score_increase = [s - baseline for s in score_progress]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Budget % vs Score Increase
    ax1 = axes[0, 0]
    ax1.plot(budget_pct, score_increase, 'b-', linewidth=2)
    ax1.scatter(budget_pct, score_increase, c='blue', s=30, alpha=0.6)
    ax1.set_xlabel('Budget (%)', fontsize=12)
    ax1.set_ylabel('Score Increase from Baseline', fontsize=12)
    ax1.set_title('Score Improvement vs Budget', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Behavioral space coverage (2D projection)
    ax2 = axes[0, 1]
    feat1 = [b['loop_count'] for b in behaviors]
    feat2 = [b['branch_count'] for b in behaviors]
    colors = scores  # Color by score
    scatter = ax2.scatter(feat1, feat2, c=colors, cmap='viridis', s=80, alpha=0.7)
    ax2.set_xlabel('Loop Count (normalized)', fontsize=12)
    ax2.set_ylabel('Branch Count (normalized)', fontsize=12)
    ax2.set_title('Behavioral Space Coverage (colored by score)', fontsize=14)
    plt.colorbar(scatter, ax=ax2, label='Score')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Another behavioral dimension pair
    ax3 = axes[1, 0]
    feat3 = [b['math_operators'] for b in behaviors]
    feat4 = [b['loop_nesting_max'] for b in behaviors]
    scatter2 = ax3.scatter(feat3, feat4, c=colors, cmap='viridis', s=80, alpha=0.7)
    ax3.set_xlabel('Math Operators (normalized)', fontsize=12)
    ax3.set_ylabel('Loop Nesting Max (normalized)', fontsize=12)
    ax3.set_title('Behavioral Space Coverage (colored by score)', fontsize=14)
    plt.colorbar(scatter2, ax=ax3, label='Score')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Score histogram
    ax4 = axes[1, 1]
    ax4.hist(scores, bins=20, edgecolor='black', alpha=0.7)
    ax4.axvline(87.4, color='red', linestyle='--', linewidth=2, label='87.4 reference')
    ax4.axvline(max(scores), color='green', linestyle='--', linewidth=2, label=f'Best: {max(scores):.2f}')
    ax4.set_xlabel('Score', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Score Distribution', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prism_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved plot to prism_analysis.png")

# Compare top solution with ~87.4 solution
print("\n=== COMPARING TOP SOLUTION VS ~87.4 SOLUTION ===")

# Find best solution
best_idx = sorted_indices[0]
best_elite = elites[best_idx]

# Find solution closest to 87.4
target_score = 87.4
closest_to_87 = min(range(len(scores)), key=lambda i: abs(scores[i] - target_score))
elite_87 = elites[closest_to_87]

print(f"Best solution: score={scores[best_idx]:.4f}")
print(f"~87.4 solution: score={scores[closest_to_87]:.4f}")

# Analyze differences
print("\nBehavioral differences:")
for feat in feature_names:
    diff = best_elite['behavior'][feat] - elite_87['behavior'][feat]
    print(f"  {feat}: {best_elite['behavior'][feat]:.4f} vs {elite_87['behavior'][feat]:.4f} (diff: {diff:+.4f})")

# Extract key algorithmic differences
def extract_algo_features(code):
    features = {}
    import re

    # Iterations
    iter_match = re.search(r'iterations?\s*=\s*(\d+)', code, re.IGNORECASE)
    features['iterations'] = int(iter_match.group(1)) if iter_match else 0

    # Restarts
    restart_match = re.search(r'for.*?(?:attempt|restart).*?range\s*\(\s*(\d+)', code, re.IGNORECASE)
    if not restart_match:
        restart_match = re.search(r'range\s*\(\s*(\d+)\s*\).*?(?:attempt|restart)', code, re.IGNORECASE)
    features['restarts'] = int(restart_match.group(1)) if restart_match else 1

    # Cooling rate
    cool_match = re.search(r'cooling_rate\s*=\s*([\d.]+)', code)
    features['cooling_rate'] = float(cool_match.group(1)) if cool_match else None

    # Temperature
    temp_match = re.search(r'temperature\s*=.*?([\d.]+)', code)
    features['initial_temp'] = float(temp_match.group(1)) if temp_match else None

    # Uses simulated annealing
    features['simulated_annealing'] = 'simulated_annealing' in code.lower()

    # Uses local search
    features['local_search'] = 'local_search' in code.lower()

    # Greedy
    features['greedy'] = 'greedy' in code.lower()

    # Sorting strategy
    if 'density' in code:
        features['sorting'] = 'density'
    elif 'size' in code and 'sorted' in code:
        features['sorting'] = 'size'
    else:
        features['sorting'] = 'other'

    return features

best_features = extract_algo_features(best_elite['code'])
elite_87_features = extract_algo_features(elite_87['code'])

print("\nAlgorithmic differences:")
for key in best_features:
    if best_features[key] != elite_87_features[key]:
        print(f"  {key}: {best_features[key]} vs {elite_87_features[key]}")

print("\n=== SUMMARY ===")
print(f"Island contains {len(elites)} solutions across {len(unique_cells)} unique cells")
print(f"Score range: {min(scores):.4f} to {max(scores):.4f} (spread: {max(scores)-min(scores):.4f})")
print(f"Best solution score: {max(scores):.4f}")
print(f"Score improvement from 87.4: {max(scores) - 87.4:.4f}")
