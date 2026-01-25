#!/usr/bin/env python3
"""
AlgoForge Diversity Analysis Script

Analyzes snapshot.json files from AlgoForge runs to assess algorithmic
and behavioral diversity.

Usage:
    python analyze_snapshot.py <snapshot_path> [--top N] [--tsne]
"""

import json
import argparse
import re
from collections import defaultdict
from pathlib import Path


def classify_paradigm(code: str) -> str:
    """Classify algorithm into paradigm family based on code patterns."""
    code_lower = code.lower()

    # Binary Search detection
    has_binary_search = False
    if any(pattern in code_lower for pattern in [
        'while lo <', 'while left <', 'while low <',
        'while lo <=', 'while left <=',
        'lo, hi', 'left, right', 'low, high'
    ]):
        if 'mid' in code_lower:
            has_binary_search = True

    # Simulated Annealing detection
    if 'temperature' in code_lower or 'temp' in code_lower:
        if 'exp(-' in code_lower or 'exp( -' in code_lower:
            if has_binary_search:
                return 'Binary Search + SA'
            return 'Simulated Annealing'

    # Genetic Algorithm detection
    if 'population' in code_lower:
        if any(x in code_lower for x in ['crossover', 'mutation', 'fitness']):
            if has_binary_search:
                return 'Binary Search + GA'
            return 'Genetic Algorithm'

    # Tabu Search detection
    if 'tabu' in code_lower:
        if has_binary_search:
            return 'Binary Search + Tabu'
        return 'Tabu Search'

    # Dynamic Programming detection
    if any(x in code_lower for x in ['memo[', 'memo =', '@lru_cache', 'dp[', 'dp =']):
        if has_binary_search:
            return 'Binary Search + DP'
        return 'Dynamic Programming'

    # Binary Search with various inner strategies
    if has_binary_search:
        # Check for bin packing patterns
        if any(x in code_lower for x in ['sorted(', 'sort(', 'first_fit', 'best_fit']):
            return 'Binary Search + Bin Packing'
        return 'Binary Search (Pure)'

    # Local Search detection
    if any(x in code_lower for x in ['improve', 'neighbor', 'hill_climb']):
        return 'Local Search'

    # Greedy patterns
    if 'sorted(' in code_lower or 'sort(' in code_lower:
        return 'Sorted Greedy'

    return 'Greedy/Heuristic'


def extract_docstring(code: str) -> str:
    """Extract docstring from code if present."""
    # Look for triple-quoted docstring
    match = re.search(r'"""(.+?)"""', code, re.DOTALL)
    if match:
        return match.group(1).strip()[:200]

    match = re.search(r"'''(.+?)'''", code, re.DOTALL)
    if match:
        return match.group(1).strip()[:200]

    return ""


def get_score(elite: dict) -> float:
    """Extract score from elite, handling different field names."""
    if 'primary_score' in elite:
        return elite['primary_score']
    if 'score' in elite:
        return elite['score']
    if 'scores' in elite and isinstance(elite['scores'], dict):
        return elite['scores'].get('score', 0)
    return 0.0


def get_cell_id(elite: dict, index: int) -> str:
    """Extract cell ID from elite, handling different field names."""
    if 'cell_index' in elite:
        return str(elite['cell_index'])
    if 'cell_id' in elite:
        return str(elite['cell_id'])
    if 'cell' in elite:
        return str(elite['cell'])
    return str(index)


def analyze_snapshot(snapshot_path: str, top_n: int = 10, show_tsne: bool = False):
    """Perform comprehensive diversity analysis on snapshot."""

    with open(snapshot_path) as f:
        snapshot = json.load(f)

    elites = snapshot.get('elites', [])

    if not elites:
        print("No elites found in snapshot!")
        return

    # Basic metrics
    n_elites = len(elites)
    cells = set(get_cell_id(e, i) for i, e in enumerate(elites))
    n_unique_cells = len(cells)

    print("=" * 60)
    print("ALGOFORGE DIVERSITY ANALYSIS")
    print("=" * 60)
    print(f"\nSnapshot: {snapshot_path}")
    print(f"\n## Archive Overview")
    print(f"- Total elites: {n_elites}")
    print(f"- Unique cells: {n_unique_cells}")

    # Sort by score
    elites_sorted = sorted(elites, key=lambda e: get_score(e), reverse=True)

    # Classify paradigms
    paradigm_counts = defaultdict(list)
    for elite in elites:
        code = elite.get('code', '')
        paradigm = classify_paradigm(code)
        paradigm_counts[paradigm].append(get_score(elite))

    print(f"\n## Paradigm Distribution")
    print("-" * 60)
    print(f"{'Paradigm':<30} {'Count':>6} {'Best':>10} {'Avg':>10}")
    print("-" * 60)

    for paradigm, scores in sorted(paradigm_counts.items(), key=lambda x: max(x[1]), reverse=True):
        print(f"{paradigm:<30} {len(scores):>6} {max(scores):>10.2f} {sum(scores)/len(scores):>10.2f}")

    print("-" * 60)

    # Top N elites
    print(f"\n## Top {top_n} Elites")
    print("-" * 70)
    print(f"{'Rank':>4} {'Score':>10} {'Cell':>6} {'Paradigm':<30}")
    print("-" * 70)

    for i, elite in enumerate(elites_sorted[:top_n], 1):
        code = elite.get('code', '')
        paradigm = classify_paradigm(code)
        cell_id = get_cell_id(elite, i)
        score = get_score(elite)
        print(f"{i:>4} {score:>10.2f} {cell_id:>6} {paradigm:<30}")

    print("-" * 70)

    # Diversity assessment
    print(f"\n## Algorithmic Diversity Assessment")

    n_paradigms = len(paradigm_counts)
    dominant = max(paradigm_counts.items(), key=lambda x: len(x[1]))
    dominant_pct = len(dominant[1]) / n_elites * 100

    if n_paradigms >= 4 and dominant_pct < 50:
        assessment = "EXCELLENT - Multiple competitive paradigm families"
    elif n_paradigms >= 3 and dominant_pct < 60:
        assessment = "GOOD - Diverse paradigms present"
    elif n_paradigms >= 2 and dominant_pct < 70:
        assessment = "MODERATE - Some diversity but one paradigm dominates"
    else:
        assessment = "LOW - Algorithmic monoculture, consider adjusting PE settings"

    print(f"\n- Paradigm families: {n_paradigms}")
    print(f"- Dominant paradigm: {dominant[0]} ({dominant_pct:.1f}%)")
    print(f"- Assessment: {assessment}")

    # Check for competitive alternatives
    if n_paradigms >= 2:
        scores_by_paradigm = [(p, max(s)) for p, s in paradigm_counts.items()]
        scores_by_paradigm.sort(key=lambda x: x[1], reverse=True)

        best_score = scores_by_paradigm[0][1]
        competitive = [(p, s) for p, s in scores_by_paradigm if s >= best_score * 0.99]

        if len(competitive) > 1:
            print(f"\n- Competitive paradigms (within 1% of best):")
            for p, s in competitive:
                print(f"  - {p}: {s:.2f}")

    # Recommendations
    print(f"\n## Recommendations")

    if n_paradigms < 3:
        print("- Enable or tune Punctuated Equilibrium to encourage paradigm shifts")
        print("- Consider using predefined centroids with diverse algorithm archetypes")

    if dominant_pct > 70:
        print("- Increase n_clusters in PE config to select more diverse representatives")
        print("- Verify PE heavy_model can generate genuinely different approaches")

    if n_unique_cells < n_elites * 0.5:
        print("- Many elites in same cells - consider increasing behavioral feature diversity")
        print("- Check if behavior_noise is needed or if features differentiate well")

    if show_tsne and n_elites >= 3:
        try:
            import numpy as np
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt

            # Extract behavior vectors
            behaviors = []
            for elite in elites:
                bv = elite.get('behavior', {})
                if isinstance(bv, dict):
                    behaviors.append(list(bv.values()))
                elif isinstance(bv, list):
                    behaviors.append(bv)

            if behaviors and all(len(b) > 0 for b in behaviors):
                X = np.array(behaviors)

                # t-SNE
                perplexity = min(30, len(X) - 1)
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                X_2d = tsne.fit_transform(X)

                # Color by paradigm
                paradigm_colors = {}
                color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                             '#ff7f00', '#ffff33', '#a65628', '#f781bf']

                for elite in elites:
                    p = classify_paradigm(elite.get('code', ''))
                    if p not in paradigm_colors:
                        paradigm_colors[p] = color_list[len(paradigm_colors) % len(color_list)]

                colors = [paradigm_colors[classify_paradigm(e.get('code', ''))] for e in elites]

                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.7, s=100)

                # Add legend
                for paradigm, color in paradigm_colors.items():
                    plt.scatter([], [], c=color, label=paradigm, s=100)
                plt.legend(loc='best', fontsize=8)

                plt.title('t-SNE of Behavioral Space (colored by paradigm)')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')

                output_path = Path(snapshot_path).parent / 'diversity_tsne.png'
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"\n- t-SNE plot saved to: {output_path}")
                plt.close()

        except ImportError as e:
            print(f"\n- t-SNE requires: pip install numpy scikit-learn matplotlib")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Analyze AlgoForge snapshot diversity')
    parser.add_argument('snapshot', help='Path to snapshot.json')
    parser.add_argument('--top', type=int, default=10, help='Number of top elites to show')
    parser.add_argument('--tsne', action='store_true', help='Generate t-SNE visualization')

    args = parser.parse_args()
    analyze_snapshot(args.snapshot, top_n=args.top, show_tsne=args.tsne)


if __name__ == '__main__':
    main()
