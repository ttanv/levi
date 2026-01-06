"""
Diversity Analysis for PRISM Problem - MAP-Elites Archive Visualization
Analyzes behavioral and algorithmic diversity in the evolved population.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

# Data extracted from snapshot.json (24 elites)
elites = [
    {"cell": 16, "score": 90.31, "loop_count": 0.611, "branch_count": 0.589, "math_operators": 0.484, "loop_nesting_max": 0.888, "algo": "Greedy+LocalSearch"},
    {"cell": 21, "score": 89.63, "loop_count": 0.554, "branch_count": 0.423, "math_operators": 0.452, "loop_nesting_max": 0.270, "algo": "Greedy+Swap"},
    {"cell": 3,  "score": 89.46, "loop_count": 0.611, "branch_count": 0.489, "math_operators": 0.405, "loop_nesting_max": 0.887, "algo": "Greedy+LocalSearch"},
    {"cell": 12, "score": 87.41, "loop_count": 0.387, "branch_count": 0.431, "math_operators": 0.506, "loop_nesting_max": 0.445, "algo": "Multi-Random+Swap"},
    {"cell": 6,  "score": 87.41, "loop_count": 0.900, "branch_count": 0.943, "math_operators": 0.955, "loop_nesting_max": 0.662, "algo": "SA+Move/Swap"},
    {"cell": 27, "score": 87.41, "loop_count": 0.525, "branch_count": 0.708, "math_operators": 0.684, "loop_nesting_max": 0.272, "algo": "SA+Restarts"},
    {"cell": 1,  "score": 87.41, "loop_count": 0.623, "branch_count": 0.731, "math_operators": 0.776, "loop_nesting_max": 0.447, "algo": "SA+Adaptive"},
    {"cell": 5,  "score": 87.38, "loop_count": 0.469, "branch_count": 0.711, "math_operators": 0.729, "loop_nesting_max": 0.273, "algo": "Greedy+SA"},
    {"cell": 15, "score": 87.37, "loop_count": 0.610, "branch_count": 0.559, "math_operators": 0.501, "loop_nesting_max": 0.887, "algo": "Greedy+Swap"},
    {"cell": 28, "score": 87.37, "loop_count": 0.775, "branch_count": 0.521, "math_operators": 0.578, "loop_nesting_max": 0.898, "algo": "Multi-Seed+Swap"},
    {"cell": 2,  "score": 86.52, "loop_count": 0.329, "branch_count": 0.393, "math_operators": 0.532, "loop_nesting_max": 0.270, "algo": "Greedy+Swap"},
    {"cell": 29, "score": 86.52, "loop_count": 0.370, "branch_count": 0.448, "math_operators": 0.597, "loop_nesting_max": 0.279, "algo": "Greedy+Move"},
    {"cell": 25, "score": 85.53, "loop_count": 0.329, "branch_count": 0.423, "math_operators": 0.637, "loop_nesting_max": 0.271, "algo": "Greedy+Heuristic"},
    {"cell": 22, "score": 85.30, "loop_count": 0.916, "branch_count": 0.480, "math_operators": 0.474, "loop_nesting_max": 0.898, "algo": "Multi-Start+Swap"},
    {"cell": 7,  "score": 82.75, "loop_count": 0.721, "branch_count": 0.609, "math_operators": 0.641, "loop_nesting_max": 0.446, "algo": "Balanced+Opt"},
    {"cell": 11, "score": 78.50, "loop_count": 0.277, "branch_count": 0.304, "math_operators": 0.387, "loop_nesting_max": 0.289, "algo": "FFD+Balance"},
    {"cell": 17, "score": 76.50, "loop_count": 0.413, "branch_count": 0.358, "math_operators": 0.333, "loop_nesting_max": 0.463, "algo": "Greedy+Simple"},
    {"cell": 9,  "score": 70.59, "loop_count": 0.396, "branch_count": 0.634, "math_operators": 0.347, "loop_nesting_max": 0.471, "algo": "BinPacking"},
    {"cell": 26, "score": 69.33, "loop_count": 0.392, "branch_count": 0.263, "math_operators": 0.278, "loop_nesting_max": 0.286, "algo": "Simple Greedy"},
    {"cell": 19, "score": 68.70, "loop_count": 0.371, "branch_count": 0.097, "math_operators": 0.526, "loop_nesting_max": 0.494, "algo": "Priority Queue"},
    {"cell": 4,  "score": 65.91, "loop_count": 0.921, "branch_count": 0.574, "math_operators": 0.715, "loop_nesting_max": 0.372, "algo": "GA+Mutation"},
    {"cell": 13, "score": 65.91, "loop_count": 0.207, "branch_count": 0.657, "math_operators": 0.483, "loop_nesting_max": 0.263, "algo": "Sorted+Fit"},
    {"cell": 14, "score": 48.15, "loop_count": 0.547, "branch_count": 0.303, "math_operators": 0.444, "loop_nesting_max": 0.278, "algo": "Basic Greedy"},
    {"cell": 0,  "score": 44.85, "loop_count": 0.227, "branch_count": 0.458, "math_operators": 0.000, "loop_nesting_max": 0.207, "algo": "Round Robin"},
]

# Create figure with custom layout
fig = plt.figure(figsize=(16, 14))
fig.suptitle('PRISM Problem: MAP-Elites Diversity Analysis\nML Model Placement Optimization', fontsize=16, fontweight='bold')

gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Color map based on score
scores = np.array([e["score"] for e in elites])
colors = plt.cm.RdYlGn((scores - scores.min()) / (scores.max() - scores.min()))

# ===== 1. Behavioral Space: Loop Count vs Branch Count =====
ax1 = fig.add_subplot(gs[0, 0])
for i, e in enumerate(elites):
    ax1.scatter(e["loop_count"], e["branch_count"], c=[colors[i]], s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
ax1.set_xlabel('Loop Count (normalized)', fontsize=10)
ax1.set_ylabel('Branch Count (normalized)', fontsize=10)
ax1.set_title('Behavioral Diversity\n(Loop vs Branch Complexity)', fontsize=11, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# ===== 2. Behavioral Space: Math Operators vs Loop Nesting =====
ax2 = fig.add_subplot(gs[0, 1])
for i, e in enumerate(elites):
    ax2.scatter(e["math_operators"], e["loop_nesting_max"], c=[colors[i]], s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
ax2.set_xlabel('Math Operators (normalized)', fontsize=10)
ax2.set_ylabel('Loop Nesting Max (normalized)', fontsize=10)
ax2.set_title('Behavioral Diversity\n(Mathematical vs Structural Complexity)', fontsize=11, fontweight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)

# ===== 3. Score Distribution =====
ax3 = fig.add_subplot(gs[0, 2])
sorted_elites = sorted(elites, key=lambda x: x["score"], reverse=True)
y_pos = np.arange(len(sorted_elites))
bar_colors = plt.cm.RdYlGn((np.array([e["score"] for e in sorted_elites]) - scores.min()) / (scores.max() - scores.min()))
bars = ax3.barh(y_pos, [e["score"] for e in sorted_elites], color=bar_colors, edgecolor='black', linewidth=0.5)
ax3.set_yticks(y_pos)
ax3.set_yticklabels([e["algo"][:12] for e in sorted_elites], fontsize=7)
ax3.set_xlabel('Score (%)', fontsize=10)
ax3.set_title('Performance Distribution\n(Score by Algorithm Type)', fontsize=11, fontweight='bold')
ax3.set_xlim(40, 95)
ax3.invert_yaxis()

# ===== 4. MAP-Elites Grid Visualization =====
ax4 = fig.add_subplot(gs[1, :2])
# Create a grid visualization (6x5 = 30 cells)
grid = np.full((6, 5), np.nan)
grid_labels = np.full((6, 5), '', dtype=object)
for e in elites:
    row = e["cell"] // 5
    col = e["cell"] % 5
    if row < 6 and col < 5:
        grid[row, col] = e["score"]
        grid_labels[row, col] = f'{e["score"]:.1f}'

# Plot heatmap
im = ax4.imshow(grid, cmap='RdYlGn', aspect='auto', vmin=40, vmax=95)
ax4.set_xticks(np.arange(5))
ax4.set_yticks(np.arange(6))
ax4.set_xlabel('Behavioral Dimension 1 (binned)', fontsize=10)
ax4.set_ylabel('Behavioral Dimension 2 (binned)', fontsize=10)
ax4.set_title('MAP-Elites Archive Grid\n(24/30 cells occupied = 80% coverage)', fontsize=11, fontweight='bold')

# Add cell annotations
for i in range(6):
    for j in range(5):
        cell_idx = i * 5 + j
        if not np.isnan(grid[i, j]):
            text_color = 'white' if grid[i, j] < 65 else 'black'
            ax4.text(j, i, f'{grid[i, j]:.0f}', ha='center', va='center', fontsize=9, color=text_color, fontweight='bold')
        else:
            ax4.text(j, i, '∅', ha='center', va='center', fontsize=12, color='gray', alpha=0.5)

plt.colorbar(im, ax=ax4, label='Score (%)', shrink=0.8)

# ===== 5. Algorithmic Diversity Analysis =====
ax5 = fig.add_subplot(gs[1, 2])
algo_categories = {
    'Greedy-based': ['Greedy+LocalSearch', 'Greedy+Swap', 'Greedy+SA', 'Greedy+Heuristic', 'Greedy+Simple', 'Simple Greedy', 'Basic Greedy'],
    'Simulated Annealing': ['SA+Move/Swap', 'SA+Restarts', 'SA+Adaptive'],
    'Multi-Start/Seed': ['Multi-Random+Swap', 'Multi-Seed+Swap', 'Multi-Start+Swap'],
    'Other Metaheuristics': ['GA+Mutation', 'BinPacking', 'Sorted+Fit', 'FFD+Balance', 'Priority Queue', 'Balanced+Opt', 'Round Robin']
}

category_scores = {}
category_counts = {}
for cat, algos in algo_categories.items():
    cat_scores = [e["score"] for e in elites if e["algo"] in algos]
    category_scores[cat] = np.mean(cat_scores) if cat_scores else 0
    category_counts[cat] = len(cat_scores)

cats = list(algo_categories.keys())
avg_scores = [category_scores[c] for c in cats]
counts = [category_counts[c] for c in cats]

x = np.arange(len(cats))
width = 0.6
bars = ax5.bar(x, avg_scores, width, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'], edgecolor='black')
ax5.set_ylabel('Average Score (%)', fontsize=10)
ax5.set_title('Algorithm Family Performance\n(Grouped by Strategy)', fontsize=11, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels([c.replace('-', '\n') for c in cats], fontsize=8)
ax5.set_ylim(50, 95)

# Add count labels
for bar, count in zip(bars, counts):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'n={count}', ha='center', fontsize=9)

# ===== 6. Parallel Coordinates for 4D Behavior =====
ax6 = fig.add_subplot(gs[2, :])
features = ['loop_count', 'branch_count', 'math_operators', 'loop_nesting_max']
feature_labels = ['Loop\nCount', 'Branch\nCount', 'Math\nOperators', 'Loop Nesting\nMax']

for i, e in enumerate(elites):
    values = [e[f] for f in features]
    alpha = 0.3 + 0.5 * (e["score"] - scores.min()) / (scores.max() - scores.min())
    ax6.plot(range(4), values, c=colors[i], alpha=alpha, linewidth=1.5, marker='o', markersize=4)

ax6.set_xticks(range(4))
ax6.set_xticklabels(feature_labels, fontsize=10)
ax6.set_ylabel('Normalized Feature Value', fontsize=10)
ax6.set_title('4D Behavioral Space - Parallel Coordinates Plot\n(Higher opacity = Better score)', fontsize=11, fontweight='bold')
ax6.set_ylim(0, 1)
ax6.grid(True, axis='y', alpha=0.3)

# Add legend for colors
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=scores.min(), vmax=scores.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax6, orientation='vertical', shrink=0.8, pad=0.02)
cbar.set_label('Score (%)', fontsize=10)

# ===== Summary Statistics Box =====
stats_text = f"""Archive Statistics
━━━━━━━━━━━━━━━━━━━━
Total Elites: 24/30 (80%)
Best Score: {scores.max():.2f}%
Worst Score: {scores.min():.2f}%
Mean Score: {scores.mean():.2f}%
Std Dev: {scores.std():.2f}%

Behavioral Diversity
━━━━━━━━━━━━━━━━━━━━
Loop Count Range: 0.21 - 0.92
Branch Count Range: 0.10 - 0.94
Math Ops Range: 0.00 - 0.96
Nesting Range: 0.21 - 0.90"""

fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout(rect=[0.0, 0.15, 1, 0.96])
plt.savefig('/Users/ttanveer/Documents/af/algoforge/examples/prism/diversity_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('/Users/ttanveer/Documents/af/algoforge/examples/prism/diversity_analysis.pdf', bbox_inches='tight')
print("Saved diversity_analysis.png and diversity_analysis.pdf")
plt.show()
