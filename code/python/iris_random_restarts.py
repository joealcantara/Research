"""
Random Restarts Experiment: Test if different initializations escape local optima

Question: Is joint search stuck at 26.8% because it always starts from empty structure?

Approach:
- Run joint search 10 times from random (structure, theory) initializations
- Track best result across all restarts
- Compare to empty init baseline (26.8%) and greedy benchmark (35.1%)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from copy import deepcopy
import random

from scripts.dag_utils import is_dag, generate_all_dags
from iris_joint_search import score_candidate, joint_beam_search


def generate_random_dag(variables, max_edges=None):
    """
    Generate a random valid DAG structure.

    Args:
        variables: List of variable names
        max_edges: Maximum number of edges (default: ~2 per variable)

    Returns:
        Dictionary mapping variables to parent lists
    """
    n = len(variables)
    if max_edges is None:
        max_edges = n * 2  # Average of 2 parents per variable

    # Generate random edges that form a DAG
    edges = set()
    max_attempts = 1000

    for _ in range(max_attempts):
        if len(edges) >= max_edges:
            break

        # Pick random edge
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)

        if i == j:
            continue

        edge = (variables[i], variables[j])
        reverse_edge = (variables[j], variables[i])

        if edge in edges or reverse_edge in edges:
            continue

        # Test if adding this edge creates a cycle
        test_edges = edges | {edge}
        var_to_idx = {var: idx for idx, var in enumerate(variables)}
        edges_idx = [(var_to_idx[p], var_to_idx[c]) for p, c in test_edges]

        if is_dag(edges_idx, n):
            edges.add(edge)

    # Convert to structure dict
    structure = {var: [] for var in variables}
    for parent, child in edges:
        structure[child].append(parent)

    return structure


def generate_random_theory(variables):
    """
    Generate random theory assignment.

    Returns:
        Dictionary mapping variables to 'linear' or 'tree'
    """
    return {var: random.choice(['linear', 'tree']) for var in variables}


# Load Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = iris.target + 1
variables = list(df.columns)

print("=" * 80)
print("Random Restarts Experiment: Escaping Local Optima")
print("=" * 80)
print()
print(f"Dataset: {len(df)} samples, {len(variables)} variables")
print(f"Strategy: Run joint search from 10 random initializations")
print(f"Baselines: Empty init (26.8%), Greedy (35.1%)")
print()

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Run multiple restarts
num_restarts = 10
beam_size = 20  # Use default beam size
max_rounds = 10

results = []

for restart_num in range(num_restarts):
    print(f"\n{'=' * 80}")
    print(f"RESTART {restart_num + 1}/{num_restarts} - Starting...")
    print(f"{'=' * 80}\n")

    # Generate random initialization
    if restart_num == 0:
        # First run: empty init (baseline)
        init_structure = {var: [] for var in variables}
        init_theory = {var: 'linear' for var in variables}
        print("Initialization: EMPTY (baseline)")
    else:
        # Random init
        init_structure = generate_random_dag(variables, max_edges=len(variables) * 2)
        init_theory = generate_random_theory(variables)

        num_edges = sum(len(parents) for parents in init_structure.values())
        num_trees = sum(1 for t in init_theory.values() if t == 'tree')

        print(f"Initialization: RANDOM")
        print(f"  Edges: {num_edges}")
        print(f"  Theory: {num_trees} trees, {len(variables) - num_trees} linear")

    # Score initial state
    init_score, init_ll = score_candidate(df, init_structure, init_theory, lambda_penalty=2.0)
    print(f"  Initial log-posterior: {init_score:.2f}")
    print()

    # Run joint beam search from this initialization
    print(f"Running joint beam search from this initialization...")
    best_structure, best_theory, best_score, history = joint_beam_search(
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
        'structure': best_structure,
        'theory': best_theory
    })

    print(f"\n✓ RESTART {restart_num + 1}/{num_restarts} COMPLETE")
    print(f"  Result: log-posterior = {best_score:.2f}")
    print(f"  Improvement from init: {best_score - init_score:.2f}")
    print()

print("\n" + "=" * 80)
print("RANDOM RESTARTS SUMMARY")
print("=" * 80)
print()

# Sort by final score
results_sorted = sorted(results, key=lambda x: -x['final_score'])

print("All restarts (sorted by final score):")
for i, r in enumerate(results_sorted):
    restart_type = "EMPTY" if r['restart'] == 0 else "RANDOM"
    print(f"  #{i+1}: Restart {r['restart']} ({restart_type})")
    print(f"      Init: {r['init_score']:.2f} → Final: {r['final_score']:.2f} (Δ {r['final_score'] - r['init_score']:+.2f})")

print()
print(f"Best log-posterior found: {results_sorted[0]['final_score']:.2f}")
print(f"Worst log-posterior found: {results_sorted[-1]['final_score']:.2f}")
print(f"Range: {results_sorted[0]['final_score'] - results_sorted[-1]['final_score']:.2f}")
print()

# Count unique final scores (to see if they converge to same optima)
unique_scores = set(r['final_score'] for r in results)
print(f"Unique final scores: {len(unique_scores)}")
if len(unique_scores) <= 3:
    print("  → Most restarts converge to same local optima")
else:
    print("  → Restarts find different local optima")
print()

# Compute posteriors for best result
print("=" * 80)
print("Computing Posterior % for Best Restart")
print("=" * 80)
print()

best_result = results_sorted[0]
best_structure = best_result['structure']
best_theory = best_result['theory']

print(f"Scoring all 29,281 structures with restart {best_result['restart']} theory assignment...")
print("(This will take a few minutes...)")
print()

all_dags = generate_all_dags(variables)
print(f"Generated {len(all_dags):,} DAG structures. Starting scoring...")

restart_results = {}
for idx, structure in enumerate(all_dags):
    if (idx + 1) % 5000 == 0:
        print(f"  Progress: {idx + 1:,} / {len(all_dags):,} structures")

    score, ll = score_candidate(df, structure, best_theory, lambda_penalty=2.0)
    restart_results[idx] = {'log_posterior': score, 'log_likelihood': ll}

# Normalize posteriors
log_posteriors = np.array([r['log_posterior'] for r in restart_results.values()])
log_posteriors_shifted = log_posteriors - log_posteriors.max()
posteriors = np.exp(log_posteriors_shifted)
posteriors = posteriors / posteriors.sum()

for i, idx in enumerate(restart_results.keys()):
    restart_results[idx]['posterior'] = posteriors[i]

print(f"\nComputed posteriors for all {len(restart_results):,} structures")
print()

# Sort and analyze
sorted_restart = sorted(restart_results.items(), key=lambda x: -x[1]['posterior'])

top_1_prob = sorted_restart[0][1]['posterior']
top_10_prob = sum(r[1]['posterior'] for r in sorted_restart[:10])
top_100_prob = sum(r[1]['posterior'] for r in sorted_restart[:100])

print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)
print()

print("Baselines:")
print("  Greedy mixed theory: 35.1%")
print("  Empty init (beam=20): 26.8%")
print()

print(f"Best Random Restart (restart {best_result['restart']}):")
print(f"  Top 1 posterior:      {top_1_prob*100:.1f}%")
print(f"  Top 10 cumulative:    {top_10_prob*100:.1f}%")
print(f"  Top 100 cumulative:   {top_100_prob*100:.1f}%")
print()

# Compare to baselines
if top_1_prob > 0.351:
    improvement = top_1_prob / 0.351
    print(f"✓ Random restart BEATS greedy! ({improvement:.2f}x)")
    print("  Different initializations escape the local basin!")
elif top_1_prob > 0.268:
    improvement = top_1_prob / 0.268
    print(f"≈ Random restart beats empty init ({improvement:.2f}x) but not greedy")
    print("  Multiple local optima exist, but greedy still better")
else:
    print("✗ Random restart stuck in same 26.8% basin")
    print("  Initialization doesn't help - local optimum is deep")

print()
print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)
