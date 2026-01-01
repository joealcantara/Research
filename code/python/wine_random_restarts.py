"""
Wine Random Restarts: Test if different initializations escape local optima

Wine has 14 variables (vs Iris 5), so search space is much larger.
We won't compute full posterior distribution (billions of structures),
just compare log-posterior scores across random restarts.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from copy import deepcopy
import random

from scripts.dag_utils import is_dag
from wine_joint_search import score_candidate, joint_beam_search


def generate_random_dag(variables, max_edges=None):
    """Generate a random valid DAG structure."""
    n = len(variables)
    if max_edges is None:
        max_edges = n * 2  # Average of 2 parents per variable

    edges = set()
    max_attempts = 2000

    for _ in range(max_attempts):
        if len(edges) >= max_edges:
            break

        i = random.randint(0, n-1)
        j = random.randint(0, n-1)

        if i == j:
            continue

        edge = (variables[i], variables[j])
        reverse_edge = (variables[j], variables[i])

        if edge in edges or reverse_edge in edges:
            continue

        test_edges = edges | {edge}
        var_to_idx = {var: idx for idx, var in enumerate(variables)}
        edges_idx = [(var_to_idx[p], var_to_idx[c]) for p, c in test_edges]

        if is_dag(edges_idx, n):
            edges.add(edge)

    structure = {var: [] for var in variables}
    for parent, child in edges:
        structure[child].append(parent)

    return structure


def generate_random_theory(variables):
    """Generate random theory assignment."""
    return {var: random.choice(['linear', 'tree']) for var in variables}


# Load Wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['class'] = wine.target + 1
variables = list(df.columns)

print("=" * 80)
print("Wine Random Restarts: Escaping Local Optima")
print("=" * 80)
print()
print(f"Dataset: {len(df)} samples, {len(variables)} variables")
print(f"Strategy: Run joint search from 10 random initializations")
print(f"Baseline: Greedy decision trees (-2977.51)")
print()

# Set random seed
random.seed(42)
np.random.seed(42)

# Run multiple restarts
num_restarts = 10
beam_size = 20
max_rounds = 10  # Wine is larger, keep rounds manageable

results = []

for restart_num in range(num_restarts):
    print(f"\n{'=' * 80}")
    print(f"RESTART {restart_num + 1}/{num_restarts} - Starting...")
    print(f"{'=' * 80}\n")

    # Generate initialization
    if restart_num == 0:
        init_structure = {var: [] for var in variables}
        init_theory = {var: 'linear' for var in variables}
        print("Initialization: EMPTY (baseline)")
    else:
        init_structure = generate_random_dag(variables, max_edges=len(variables) * 2)
        init_theory = generate_random_theory(variables)

        num_edges = sum(len(parents) for parents in init_structure.values())
        num_trees = sum(1 for t in init_theory.values() if t == 'tree')

        print(f"Initialization: RANDOM")
        print(f"  Edges: {num_edges}")
        print(f"  Theory: {num_trees} trees, {len(variables) - num_trees} linear")

    init_score, init_ll = score_candidate(df, init_structure, init_theory, lambda_penalty=2.0)
    print(f"  Initial log-posterior: {init_score:.2f}")
    print()

    # Run joint search
    print(f"Running joint beam search from this initialization...")
    best_structure, best_theory, best_score, best_ll = joint_beam_search(
        df, variables,
        beam_size=beam_size,
        max_rounds=max_rounds,
        lambda_penalty=2.0,
        init_structure=init_structure,
        init_theory=init_theory
    )

    # Store result
    results.append({
        'restart': restart_num,
        'init_score': init_score,
        'final_score': best_score,
        'improvement': best_score - init_score
    })

    print(f"\n✓ RESTART {restart_num + 1}/{num_restarts} COMPLETE")
    print(f"  Result: log-posterior = {best_score:.2f}")
    print(f"  Improvement from init: {best_score - init_score:.2f}")
    print()

print("\n" + "=" * 80)
print("WINE RANDOM RESTARTS SUMMARY")
print("=" * 80)
print()

# Sort by final score
results_sorted = sorted(results, key=lambda x: -x['final_score'])

print("All restarts (sorted by final score):")
for i, r in enumerate(results_sorted):
    restart_type = "EMPTY" if r['restart'] == 0 else "RANDOM"
    print(f"  #{i+1}: Restart {r['restart']} ({restart_type})")
    print(f"      Init: {r['init_score']:.2f} → Final: {r['final_score']:.2f} (Δ {r['improvement']:+.2f})")

print()
print(f"Best log-posterior found: {results_sorted[0]['final_score']:.2f}")
print(f"Worst log-posterior found: {results_sorted[-1]['final_score']:.2f}")
print(f"Range: {results_sorted[0]['final_score'] - results_sorted[-1]['final_score']:.2f}")
print()

# Count unique final scores
unique_scores = set(round(r['final_score'], 1) for r in results)  # Round to 0.1 for grouping
print(f"Unique final scores (±0.1): {len(unique_scores)}")
if len(unique_scores) <= 3:
    print("  → Most restarts converge to same local optima")
else:
    print("  → Restarts find different local optima")
print()

print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)
print()

print("Baseline:")
print("  Greedy decision trees: -2977.51")
print()

print(f"Best Random Restart (restart {results_sorted[0]['restart']}):")
print(f"  Log-posterior: {results_sorted[0]['final_score']:.2f}")
print()

# Compare to baseline
best_score = results_sorted[0]['final_score']
if best_score > -2977.51:
    print(f"✓ Random restart BEATS greedy!")
    print(f"  Different initializations escape the local basin!")
elif best_score > -3555.63:  # Previous joint search result
    print(f"≈ Random restart better than empty init but not greedy")
    print(f"  Multiple local optima exist, but greedy still better")
else:
    print("✗ Random restart stuck in same or worse basin")
    print("  Initialization doesn't help on Wine either")

print()
print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)
