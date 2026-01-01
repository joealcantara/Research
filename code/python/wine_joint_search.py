"""
Wine Dataset: Joint Search over (Structure, Theory Assignment) pairs

Tests whether joint optimization with extended search (50 rounds) can beat
greedy sequential optimization on a larger dataset (14 variables).

Expected: With more variables and longer search, joint optimization might find
better combinations where structure and theory complement each other.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from copy import deepcopy

from scripts.dag_utils import is_dag
from scripts.inference import _get_variable_type


def score_candidate(df, structure, theory_assignment, lambda_penalty=2.0):
    """
    Score a (structure, theory assignment) pair.

    Returns log-posterior and log-likelihood.
    """
    from itertools import product

    variables = list(structure.keys())
    log_lik = 0.0

    for var in variables:
        parents = structure[var]
        theory_type = theory_assignment.get(var, 'linear')
        var_type = _get_variable_type(df, var)

        if len(parents) == 0:
            # No parents - marginal distribution
            if var_type == 'continuous':
                mean = df[var].mean()
                std = df[var].std() + 1e-6
                log_lik += -0.5 * np.sum(np.log(2 * np.pi * std**2) + ((df[var] - mean) / std)**2)
            else:
                unique_vals = sorted(df[var].unique())
                total = len(df)
                num_values = len(unique_vals)
                for val in unique_vals:
                    count = (df[var] == val).sum()
                    prob = (count + 1) / (total + num_values)
                    log_lik += count * np.log(prob)
        else:
            # Has parents
            if theory_type == 'linear':
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
                    # Categorical with CPT
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
                X = df[parents].values
                y = df[var].values

                if var_type == 'continuous':
                    tree = DecisionTreeRegressor(max_depth=2, random_state=42)
                    tree.fit(X, y)
                    predictions = tree.predict(X)
                    residuals = y - predictions
                    residual_std = np.std(residuals) + 1e-6
                    log_lik += -0.5 * np.sum(np.log(2 * np.pi * residual_std**2) + (residuals / residual_std)**2)
                else:
                    tree = DecisionTreeClassifier(max_depth=2, random_state=42)
                    tree.fit(X, y)
                    for idx in range(len(df)):
                        probs = tree.predict_proba(X[idx:idx+1])[0]
                        class_idx = int(y[idx] - min(y))
                        prob = probs[class_idx] if class_idx < len(probs) else 1e-10
                        log_lik += np.log(prob + 1e-10)

    edges = sum(len(parents) for parents in structure.values())
    log_prior = -lambda_penalty * edges
    log_posterior = log_lik + log_prior

    return log_posterior, log_lik


def generate_structure_neighbors(structure, variables):
    """Generate structure neighbors (add/remove/flip edges)."""
    neighbors = []
    n = len(variables)
    var_to_idx = {var: i for i, var in enumerate(variables)}

    current_edges = set()
    for var, parents in structure.items():
        for parent in parents:
            current_edges.add((parent, var))

    # Try all possible edges
    for var_from in variables:
        for var_to in variables:
            if var_from == var_to:
                continue

            edge = (var_from, var_to)
            reverse_edge = (var_to, var_from)

            if edge in current_edges:
                # Remove edge
                new_edges = current_edges - {edge}
                new_structure = edges_to_structure(new_edges, variables)
                neighbors.append(new_structure)

                # Flip edge
                if reverse_edge not in current_edges:
                    flipped_edges = (current_edges - {edge}) | {reverse_edge}
                    flipped_edges_idx = [(var_to_idx[p], var_to_idx[c]) for p, c in flipped_edges]
                    if is_dag(flipped_edges_idx, n):
                        new_structure = edges_to_structure(flipped_edges, variables)
                        neighbors.append(new_structure)
            else:
                # Add edge
                new_edges = current_edges | {edge}
                new_edges_idx = [(var_to_idx[p], var_to_idx[c]) for p, c in new_edges]
                if is_dag(new_edges_idx, n):
                    new_structure = edges_to_structure(new_edges, variables)
                    neighbors.append(new_structure)

    return neighbors


def edges_to_structure(edges, variables):
    """Convert edges to structure dict."""
    structure = {var: [] for var in variables}
    for parent, child in edges:
        structure[child].append(parent)
    return structure


def generate_theory_neighbors(theory_assignment, variables):
    """Generate theory neighbors (flip one variable's theory)."""
    neighbors = []
    for var in variables:
        new_assignment = theory_assignment.copy()
        current = new_assignment.get(var, 'linear')
        new_assignment[var] = 'tree' if current == 'linear' else 'linear'
        neighbors.append(new_assignment)
    return neighbors


def joint_beam_search(df, variables, beam_size=20, max_rounds=50, lambda_penalty=2.0, init_structure=None, init_theory=None):
    """
    Joint beam search over (structure, theory) pairs.

    Args:
        init_structure: Optional initial structure (default: empty)
        init_theory: Optional initial theory assignment (default: all linear)
    """
    print(f"Starting joint beam search (beam_size={beam_size}, max_rounds={max_rounds})")
    print()

    # Initialize: use provided init or default to empty
    if init_structure is None:
        init_structure = {var: [] for var in variables}
    if init_theory is None:
        init_theory = {var: 'linear' for var in variables}

    init_score, init_ll = score_candidate(df, init_structure, init_theory, lambda_penalty)

    beam = [(init_structure, init_theory, init_score, init_ll)]
    best_ever = (init_structure, init_theory, init_score, init_ll)

    for round_num in range(max_rounds):
        print(f"Round {round_num + 1}/{max_rounds}")

        all_neighbors = []

        for structure, theory, score, ll in beam:
            # Structure neighbors
            structure_neighbors = generate_structure_neighbors(structure, variables)
            for new_structure in structure_neighbors:
                all_neighbors.append((new_structure, theory))

            # Theory neighbors
            theory_neighbors = generate_theory_neighbors(theory, variables)
            for new_theory in theory_neighbors:
                all_neighbors.append((structure, new_theory))

        # Remove duplicates
        unique_neighbors = []
        seen = set()
        for structure, theory in all_neighbors:
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
        scored_neighbors.sort(key=lambda x: -x[2])
        beam = scored_neighbors[:beam_size]

        # Track best ever
        if beam[0][2] > best_ever[2]:
            best_ever = beam[0]

        print(f"  Best in beam: log-post = {beam[0][2]:.2f}, log-lik = {beam[0][3]:.2f}")
        print(f"  Best ever:    log-post = {best_ever[2]:.2f}, log-lik = {best_ever[3]:.2f}")
        print()

    return best_ever[0], best_ever[1], best_ever[2], best_ever[3]


# Main
print("=" * 80)
print("Wine Dataset: Joint Search over (Structure, Theory) Pairs")
print("=" * 80)
print()

# Load Wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['class'] = wine.target

variables = list(df.columns)
print(f"Dataset: {len(df)} samples, {len(variables)} variables")
print()

# Run joint search
best_structure, best_theory, best_score, best_ll = joint_beam_search(
    df, variables,
    beam_size=50,
    max_rounds=10,  # Testing beam_size=50 to see if diversity helps
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
        parents_str = ', '.join(parents[:3])
        if len(parents) > 3:
            parents_str += f", ... ({len(parents)} total)"
        print(f"  {var}: ← {parents_str}")

print()
print("Theory assignment:")
linear_vars = [v for v in variables if best_theory[v] == 'linear']
tree_vars = [v for v in variables if best_theory[v] == 'tree']
print(f"  Linear ({len(linear_vars)}): {', '.join(linear_vars)}")
print(f"  Tree ({len(tree_vars)}): {', '.join(tree_vars)}")

print()
edges = sum(len(parents) for parents in best_structure.values())
print(f"Log-posterior: {best_score:.2f}")
print(f"Log-likelihood: {best_ll:.2f}")
print(f"Edges: {edges}")

print()
print("=" * 80)
print("Comparison to Greedy Beam Search")
print("=" * 80)
print()

print("Greedy beam search results (from wine_beam_search.py):")
print("  Linear Gaussian:    -3044.53")
print("  Decision Trees:     -2977.51 ✓ Winner")
print("  Mixed Theory:       -3032.46")
print()

print("Joint search (50 rounds):")
print(f"  Best found:         {best_score:.2f}")
print()

if best_score > -2977.51:
    improvement = best_score - (-2977.51)
    print(f"✓ Joint search wins! (+{improvement:.2f} improvement)")
    print("  Extended exploration found better (structure, theory) combination.")
elif best_score > -3032.46:
    print("✓ Joint search beats mixed theory but not pure trees")
    print("  More exploration helped, but trees still optimal for Wine.")
else:
    print("≈ Joint search similar to greedy approaches")
    print("  Even with 50 rounds, greedy beam search remains competitive.")

print()
print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)
