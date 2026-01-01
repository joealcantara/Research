"""
Option 2: Joint Search over (Structure, Theory Assignment) pairs

Instead of first finding the best structure, then assigning theories, we search
over the joint space of (DAG structure, theory assignment) pairs using beam search.

This should find better combinations than the greedy approach since we're optimizing
both structure and theory simultaneously.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from copy import deepcopy

from scripts.dag_utils import is_dag, count_edges
from scripts.inference import _get_variable_type


def score_candidate(df, structure, theory_assignment, lambda_penalty=2.0):
    """
    Score a (structure, theory assignment) pair.

    Returns log-posterior and log-likelihood.
    """
    variables = list(structure.keys())

    # Score each variable with its assigned theory
    log_lik = 0.0
    for var in variables:
        parents = structure[var]
        theory_type = theory_assignment.get(var, 'linear')
        var_type = _get_variable_type(df, var)

        if len(parents) == 0:
            # No parents - marginal distribution (same for both theories)
            if var_type == 'continuous':
                mean = df[var].mean()
                std = df[var].std() + 1e-6
                log_lik += -0.5 * np.sum(np.log(2 * np.pi * std**2) + ((df[var] - mean) / std)**2)
            else:
                # Categorical marginal
                unique_vals = sorted(df[var].unique())
                total = len(df)
                num_values = len(unique_vals)
                for val in unique_vals:
                    count = (df[var] == val).sum()
                    prob = (count + 1) / (total + num_values)
                    log_lik += count * np.log(prob)
        else:
            # Has parents - use assigned theory
            if theory_type == 'linear':
                # Linear regression (for continuous) or CPT (for categorical)
                if var_type == 'continuous':
                    X = df[parents].values
                    y = df[var].values
                    X_with_intercept = np.column_stack([np.ones(len(X)), X])
                    try:
                        coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                        predictions = X_with_intercept @ coeffs
                        residuals = y - predictions
                        residual_std = np.std(residuals) + 1e-6
                        log_lik += -0.5 * np.sum(np.log(2 * np.pi * residual_std**2) + (residuals / residual_std)**2)
                    except:
                        log_lik += -np.inf
                else:
                    # Categorical with CPT (discretize continuous parents)
                    from itertools import product
                    parent_values = []
                    for p in parents:
                        p_type = _get_variable_type(df, p)
                        if p_type == 'continuous':
                            parent_values.append([0, 1])
                        else:
                            parent_values.append(sorted(df[p].unique()))

                    all_combos = list(product(*parent_values))
                    unique_vals = sorted(df[var].unique())
                    num_values = len(unique_vals)

                    for combo in all_combos:
                        mask = pd.Series([True] * len(df))
                        for i, parent in enumerate(parents):
                            p_type = _get_variable_type(df, parent)
                            if p_type == 'continuous':
                                median = df[parent].median()
                                if combo[i] == 0:
                                    mask = mask & (df[parent] <= median)
                                else:
                                    mask = mask & (df[parent] > median)
                            else:
                                mask = mask & (df[parent] == combo[i])

                        subset = df[mask]
                        if len(subset) == 0:
                            continue

                        total = len(subset)
                        for val in unique_vals:
                            count = (subset[var] == val).sum()
                            prob = (count + 1) / (total + num_values)
                            log_lik += count * np.log(prob)

            elif theory_type == 'tree':
                # Decision tree
                X = df[parents].values
                y = df[var].values

                if var_type == 'continuous':
                    # Regression tree
                    tree = DecisionTreeRegressor(max_depth=2, random_state=42)
                    tree.fit(X, y)
                    predictions = tree.predict(X)
                    residuals = y - predictions
                    residual_std = np.std(residuals) + 1e-6
                    log_lik += -0.5 * np.sum(np.log(2 * np.pi * residual_std**2) + (residuals / residual_std)**2)
                else:
                    # Classification tree
                    tree = DecisionTreeClassifier(max_depth=2, random_state=42)
                    tree.fit(X, y)
                    for idx in range(len(df)):
                        probs = tree.predict_proba(X[idx:idx+1])[0]
                        class_idx = int(y[idx] - min(y))
                        prob = probs[class_idx] if class_idx < len(probs) else 1e-10
                        log_lik += np.log(prob + 1e-10)

    # Compute posterior
    edges = sum(len(parents) for parents in structure.values())
    log_prior = -lambda_penalty * edges
    log_posterior = log_lik + log_prior

    return log_posterior, log_lik


def generate_structure_neighbors(structure, variables):
    """
    Generate all valid DAG structures that differ by one edge from current structure.

    Operations: add edge, remove edge, flip edge direction
    """
    neighbors = []
    n = len(variables)

    # Create mapping between variables and indices
    var_to_idx = {var: i for i, var in enumerate(variables)}
    idx_to_var = {i: var for i, var in enumerate(variables)}

    current_edges = set()
    for var, parents in structure.items():
        for parent in parents:
            current_edges.add((parent, var))

    # Try all possible edges
    for i, var_from in enumerate(variables):
        for j, var_to in enumerate(variables):
            if i == j:
                continue

            edge = (var_from, var_to)
            reverse_edge = (var_to, var_from)

            if edge in current_edges:
                # Try removing this edge
                new_edges = current_edges - {edge}
                new_structure = edges_to_structure(new_edges, variables)
                neighbors.append(new_structure)

                # Try flipping this edge
                if reverse_edge not in current_edges:
                    flipped_edges = (current_edges - {edge}) | {reverse_edge}
                    # Convert to integer indices for is_dag check
                    flipped_edges_idx = [(var_to_idx[p], var_to_idx[c]) for p, c in flipped_edges]
                    if is_dag(flipped_edges_idx, n):
                        new_structure = edges_to_structure(flipped_edges, variables)
                        neighbors.append(new_structure)
            else:
                # Try adding this edge
                new_edges = current_edges | {edge}
                # Convert to integer indices for is_dag check
                new_edges_idx = [(var_to_idx[p], var_to_idx[c]) for p, c in new_edges]
                if is_dag(new_edges_idx, n):
                    new_structure = edges_to_structure(new_edges, variables)
                    neighbors.append(new_structure)

    return neighbors


def edges_to_structure(edges, variables):
    """Convert set of edges to structure dict."""
    structure = {var: [] for var in variables}
    for parent, child in edges:
        structure[child].append(parent)
    return structure


def generate_theory_neighbors(theory_assignment, variables):
    """
    Generate all theory assignments that differ by flipping one variable's theory.
    """
    neighbors = []

    for var in variables:
        new_assignment = theory_assignment.copy()
        # Flip this variable's theory
        current = new_assignment.get(var, 'linear')
        new_assignment[var] = 'tree' if current == 'linear' else 'linear'
        neighbors.append(new_assignment)

    return neighbors


def joint_beam_search(df, variables, beam_size=20, max_rounds=5, lambda_penalty=2.0, init_structure=None, init_theory=None):
    """
    Beam search over (structure, theory assignment) pairs.

    Args:
        df: DataFrame
        variables: List of variable names
        beam_size: Number of candidates to keep in beam
        max_rounds: Number of search iterations
        lambda_penalty: Complexity penalty per edge
        init_structure: Optional initial structure (default: empty)
        init_theory: Optional initial theory assignment (default: all linear)

    Returns:
        (best_structure, best_theory, best_score, search_history)
    """
    print(f"Starting joint beam search (beam_size={beam_size}, max_rounds={max_rounds})")
    print()

    # Initialize: use provided init or default to empty structure and all linear theories
    if init_structure is None:
        init_structure = {var: [] for var in variables}
    if init_theory is None:
        init_theory = {var: 'linear' for var in variables}

    init_score, init_ll = score_candidate(df, init_structure, init_theory, lambda_penalty)

    beam = [(init_structure, init_theory, init_score, init_ll)]
    best_ever = (init_structure, init_theory, init_score, init_ll)

    search_history = []

    for round_num in range(max_rounds):
        print(f"Round {round_num + 1}/{max_rounds}")
        print("-" * 80)

        # Generate all neighbors
        all_neighbors = []

        for structure, theory, score, ll in beam:
            # Structure neighbors (keep theory fixed)
            structure_neighbors = generate_structure_neighbors(structure, variables)
            for new_structure in structure_neighbors:
                all_neighbors.append((new_structure, theory))

            # Theory neighbors (keep structure fixed)
            theory_neighbors = generate_theory_neighbors(theory, variables)
            for new_theory in theory_neighbors:
                all_neighbors.append((structure, new_theory))

        # Remove duplicates
        unique_neighbors = []
        seen = set()
        for structure, theory in all_neighbors:
            # Create hashable representation
            structure_key = tuple(sorted((var, tuple(sorted(parents))) for var, parents in structure.items()))
            theory_key = tuple(sorted(theory.items()))
            key = (structure_key, theory_key)

            if key not in seen:
                seen.add(key)
                unique_neighbors.append((structure, theory))

        print(f"  Generated {len(unique_neighbors)} unique neighbors")

        # Score all neighbors
        scored_neighbors = []
        for structure, theory in unique_neighbors:
            score, ll = score_candidate(df, structure, theory, lambda_penalty)
            scored_neighbors.append((structure, theory, score, ll))

        # Keep top beam_size
        scored_neighbors.sort(key=lambda x: -x[2])  # Sort by score (descending)
        beam = scored_neighbors[:beam_size]

        # Track best ever seen
        if beam[0][2] > best_ever[2]:
            best_ever = beam[0]

        # Record history
        search_history.append({
            'round': round_num + 1,
            'best_score': beam[0][2],
            'best_ll': beam[0][3],
            'beam_scores': [x[2] for x in beam[:5]]
        })

        print(f"  Best in beam: log-posterior = {beam[0][2]:.2f}, log-likelihood = {beam[0][3]:.2f}")
        print(f"  Best ever:    log-posterior = {best_ever[2]:.2f}, log-likelihood = {best_ever[3]:.2f}")
        print()

    return best_ever[0], best_ever[1], best_ever[2], search_history


# Main benchmark
print("=" * 80)
print("Joint Search: (Structure, Theory Assignment) Optimization")
print("=" * 80)
print()

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = iris.target + 1

variables = list(df.columns)
print(f"Dataset: {len(df)} samples, {len(variables)} variables")
print(f"Variables: {', '.join(variables)}")
print()

# Run joint beam search
best_structure, best_theory, best_score, history = joint_beam_search(
    df, variables,
    beam_size=100,
    max_rounds=10,
    lambda_penalty=2.0
)

print("=" * 80)
print("Search Complete")
print("=" * 80)
print()

print("Best (structure, theory) pair found:")
print()
print("Structure:")
for var in variables:
    parents = best_structure[var]
    if len(parents) == 0:
        print(f"  {var}: (no parents)")
    else:
        print(f"  {var}: ← {', '.join(parents)}")

print()
print("Theory assignment:")
for var in variables:
    theory = best_theory[var]
    print(f"  {var}: {theory}")

print()
print(f"Log-posterior: {best_score:.2f}")
edges = sum(len(parents) for parents in best_structure.values())
print(f"Edges: {edges}")

# Compute posterior probability (need to compare to baselines)
print()
print("=" * 80)
print("Comparison to Previous Approaches")
print("=" * 80)
print()

# Load baseline scores from previous benchmarks
print("Previous results:")
print("  Binary discretization:  3.3% top posterior")
print("  Linear Gaussian:       17.5% top posterior")
print("  Mixed theory (greedy): 35.1% top posterior")
print()

print("Joint search result:")
print(f"  Log-posterior: {best_score:.2f}")
print()

# To properly compare, we'd need to compute posteriors across all structures
# For now, just report the log-posterior improvement
print("Note: To compute comparable posterior %, we'd need to score all 29,281 structures")
print("with the found theory assignment and normalize. The log-posterior directly shows")
print("the improvement in model fit.")
print()

print("Search trajectory:")
for record in history:
    print(f"  Round {record['round']}: best = {record['best_score']:.2f} (ll = {record['best_ll']:.2f})")

print()
print("=" * 80)
print("Computing Full Posterior Distribution")
print("=" * 80)
print()

# Now score ALL structures with the found theory assignment
print("Scoring all 29,281 structures with joint search theory assignment...")
print("(This will take a few minutes)")
print()

from scripts.dag_utils import generate_all_dags

all_dags = generate_all_dags(variables)

joint_results = {}
for idx, structure in enumerate(all_dags):
    if (idx + 1) % 5000 == 0:
        print(f"  Progress: {idx + 1:,} / {len(all_dags):,} structures")

    score, ll = score_candidate(df, structure, best_theory, lambda_penalty=2.0)
    joint_results[idx] = {'log_posterior': score, 'log_likelihood': ll}

# Normalize posteriors
log_posteriors = np.array([r['log_posterior'] for r in joint_results.values()])
log_posteriors_shifted = log_posteriors - log_posteriors.max()
posteriors = np.exp(log_posteriors_shifted)
posteriors = posteriors / posteriors.sum()

for i, idx in enumerate(joint_results.keys()):
    joint_results[idx]['posterior'] = posteriors[i]

print(f"\nComputed posteriors for all {len(joint_results):,} structures")
print()

# Sort and analyze
sorted_joint = sorted(joint_results.items(), key=lambda x: -x[1]['posterior'])

top_1_prob = sorted_joint[0][1]['posterior']
top_10_prob = sum(r[1]['posterior'] for r in sorted_joint[:10])
top_100_prob = sum(r[1]['posterior'] for r in sorted_joint[:100])

print("=" * 80)
print("FINAL COMPARISON")
print("=" * 80)
print()

print("Binary discretization:")
print("  Top 1 posterior:        3.3%")
print("  Top 10 cumulative:     23.5%")
print("  Top 100 cumulative:    60.2%")
print()

print("Linear Gaussian:")
print("  Top 1 posterior:       17.5%")
print("  Top 10 cumulative:     76.3%")
print("  Top 100 cumulative:   100.0%")
print()

print("Mixed Theory (greedy):")
print("  Top 1 posterior:       35.1%")
print("  Top 10 cumulative:     95.7%")
print("  Top 100 cumulative:   100.0%")
print()

print("Joint Search (this run):")
print(f"  Top 1 posterior:       {top_1_prob*100:.1f}%")
print(f"  Top 10 cumulative:     {top_10_prob*100:.1f}%")
print(f"  Top 100 cumulative:    {top_100_prob*100:.1f}%")
print()

# Compute improvement
improvement = top_1_prob / 0.351  # vs greedy mixed theory
print(f"Improvement over greedy mixed theory: {improvement:.2f}x")
print()

if top_1_prob > 0.351 * 1.05:
    print("✓ Joint search provides better structure identification!")
    print("  Optimizing structure and theory together beats greedy sequential optimization.")
elif top_1_prob < 0.351 * 0.95:
    print("✗ Joint search performed worse than greedy approach")
    print("  May need more search rounds or different beam size.")
else:
    print("≈ Similar performance to greedy mixed theory")
    print("  Joint search found comparable solution through different path.")

print()
print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)
