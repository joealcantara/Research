"""
Wine Dataset: Beam Search for Structure Learning with Theory Languages

With 14 variables, we cannot enumerate all possible DAGs (billions+).
Instead, we use beam search to find high-scoring structures and compare
different theory language assignments.

Dataset: 178 samples, 13 features + 1 class = 14 variables
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

from scripts.inference import _get_variable_type


def custom_beam_search(df, target, score_fn, k=10, max_rounds=3, verbose=False):
    """
    Custom beam search with user-provided scoring function.

    Args:
        df: DataFrame
        target: Target variable
        score_fn: Function that takes parent_vars list and returns score
        k: Beam width
        max_rounds: Maximum rounds
        verbose: Print progress

    Returns:
        List of (theory, score) tuples sorted by score
    """
    predictors = [c for c in df.columns if c != target]

    # Initialize with single-variable theories + empty theory
    theories = [[v] for v in predictors] + [[]]
    scored = [(t, score_fn(t)) for t in theories]

    # Keep top k
    scored.sort(key=lambda x: -x[1])
    current = scored[:k]
    best_score = current[0][1]

    for round_num in range(max_rounds):
        candidates = []

        for theory, old_score in current:
            # Keep original
            candidates.append((theory, old_score))

            # Try adding each predictor
            for v in predictors:
                if v not in theory:
                    new_theory = theory + [v]
                    new_score = score_fn(new_theory)
                    candidates.append((new_theory, new_score))

        # Remove duplicates
        seen = set()
        unique_candidates = []
        for theory, score in candidates:
            key = tuple(sorted(theory))
            if key not in seen:
                seen.add(key)
                unique_candidates.append((theory, score))

        # Keep top k
        unique_candidates.sort(key=lambda x: -x[1])
        current = unique_candidates[:k]

        new_best = current[0][1]

        if verbose:
            print(f"  Round {round_num + 1}: best score = {new_best:.2f}")

        # Early stopping
        if new_best <= best_score + 0.1:
            break

        best_score = new_best

    return current


def score_theory_with_type(df, parent_vars, target, theory_type='linear', lambda_penalty=2.0):
    """
    Score a theory (parent set → target) using specified theory type.

    Returns log-posterior score for this variable.
    """
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from itertools import product

    var_type = _get_variable_type(df, target)

    if len(parent_vars) == 0:
        # No parents - marginal distribution (same for both theories)
        if var_type == 'continuous':
            mean = df[target].mean()
            std = df[target].std() + 1e-6
            log_lik = -0.5 * np.sum(np.log(2 * np.pi * std**2) + ((df[target] - mean) / std)**2)
        else:
            # Categorical marginal
            unique_vals = sorted(df[target].unique())
            total = len(df)
            num_values = len(unique_vals)
            log_lik = 0.0
            for val in unique_vals:
                count = (df[target] == val).sum()
                prob = (count + 1) / (total + num_values)
                log_lik += count * np.log(prob)

        log_prior = 0  # No edges
        return log_lik + log_prior

    # Has parents
    if theory_type == 'linear':
        if var_type == 'continuous':
            # Linear regression
            X = df[parent_vars].values
            y = df[target].values
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            try:
                coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                predictions = X_with_intercept @ coeffs
                residuals = y - predictions
                residual_std = np.std(residuals) + 1e-6
                log_lik = -0.5 * np.sum(np.log(2 * np.pi * residual_std**2) + (residuals / residual_std)**2)
            except:
                log_lik = -np.inf
        else:
            # Categorical with CPT
            parent_values = []
            for p in parent_vars:
                p_type = _get_variable_type(df, p)
                if p_type == 'continuous':
                    parent_values.append([0, 1])
                else:
                    parent_values.append(sorted(df[p].unique()))

            all_combos = list(product(*parent_values))
            unique_vals = sorted(df[target].unique())
            num_values = len(unique_vals)

            log_lik = 0.0
            for combo in all_combos:
                mask = pd.Series([True] * len(df))
                for i, parent in enumerate(parent_vars):
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
                    count = (subset[target] == val).sum()
                    prob = (count + 1) / (total + num_values)
                    log_lik += count * np.log(prob)

    elif theory_type == 'tree':
        # Decision tree
        X = df[parent_vars].values
        y = df[target].values

        if var_type == 'continuous':
            tree = DecisionTreeRegressor(max_depth=2, random_state=42)
            tree.fit(X, y)
            predictions = tree.predict(X)
            residuals = y - predictions
            residual_std = np.std(residuals) + 1e-6
            log_lik = -0.5 * np.sum(np.log(2 * np.pi * residual_std**2) + (residuals / residual_std)**2)
        else:
            tree = DecisionTreeClassifier(max_depth=2, random_state=42)
            tree.fit(X, y)
            log_lik = 0.0
            for idx in range(len(df)):
                probs = tree.predict_proba(X[idx:idx+1])[0]
                class_idx = int(y[idx] - min(y))
                prob = probs[class_idx] if class_idx < len(probs) else 1e-10
                log_lik += np.log(prob + 1e-10)

    # Apply complexity penalty
    log_prior = -lambda_penalty * len(parent_vars)
    return log_lik + log_prior


def beam_search_structure(df, variables, theory_assignment, beam_size=10, max_rounds=5, lambda_penalty=2.0):
    """
    Use beam search to find a good DAG structure given a theory assignment.

    For each variable, find high-scoring parent sets using beam search.
    """
    structure = {}
    total_score = 0.0

    print(f"Running beam search for each variable (beam_size={beam_size}, max_rounds={max_rounds})...")
    print()

    for var in variables:
        theory_type = theory_assignment.get(var, 'linear')

        # Find best parent set for this variable
        candidates = [v for v in variables if v != var]

        # Custom scoring function with theory type
        def score_fn(parent_vars):
            return score_theory_with_type(df, parent_vars, var, theory_type, lambda_penalty)

        # Run beam search
        results = custom_beam_search(
            df, var, score_fn=score_fn,
            k=beam_size, max_rounds=max_rounds,
            verbose=False
        )

        # Take best result
        if len(results) > 0:
            best_parents, best_score = results[0]
            structure[var] = list(best_parents)
            total_score += best_score

            if len(best_parents) == 0:
                print(f"  {var}: (no parents) | score: {best_score:.2f} | theory: {theory_type}")
            else:
                print(f"  {var}: ← {', '.join(best_parents)} | score: {best_score:.2f} | theory: {theory_type}")
        else:
            structure[var] = []
            print(f"  {var}: (no parents - no results)")

    print()
    print(f"Total structure score: {total_score:.2f}")
    print()

    return structure, total_score


# Load Wine dataset
print("=" * 80)
print("Wine Dataset: Beam Search Structure Learning")
print("=" * 80)
print()

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['class'] = wine.target  # 0, 1, 2

print(f"Dataset: {len(df)} samples, {len(df.columns)} variables")
print()
print("Variables:")
for i, col in enumerate(df.columns, 1):
    if col == 'class':
        print(f"  {i:2d}. {col:<25s} - categorical (3 wine types)")
    else:
        print(f"  {i:2d}. {col:<25s} - continuous (mean={df[col].mean():.2f}, std={df[col].std():.2f})")
print()

variables = list(df.columns)

# Approach 1: Pure Linear Gaussian
print("=" * 80)
print("Approach 1: Pure Linear Gaussian")
print("=" * 80)
print()

theory_linear = {var: 'linear' for var in variables}
structure_linear, score_linear = beam_search_structure(
    df, variables, theory_linear,
    beam_size=10, max_rounds=5, lambda_penalty=2.0
)

# Approach 2: Pure Decision Trees
print("=" * 80)
print("Approach 2: Pure Decision Trees (depth=2)")
print("=" * 80)
print()

theory_tree = {var: 'tree' for var in variables}
structure_tree, score_tree = beam_search_structure(
    df, variables, theory_tree,
    beam_size=10, max_rounds=5, lambda_penalty=2.0
)

# Approach 3: Mixed Theory (categorical → tree, continuous → linear)
print("=" * 80)
print("Approach 3: Mixed Theory Assignment")
print("=" * 80)
print()

theory_mixed = {}
for var in variables:
    var_type = _get_variable_type(df, var)
    if var_type == 'categorical':
        theory_mixed[var] = 'tree'
        print(f"  {var}: tree (categorical classification)")
    else:
        theory_mixed[var] = 'linear'
        print(f"  {var}: linear (continuous)")

print()

structure_mixed, score_mixed = beam_search_structure(
    df, variables, theory_mixed,
    beam_size=10, max_rounds=5, lambda_penalty=2.0
)

# Summary
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print()

print(f"{'Approach':<30s} | {'Total Score':<15s} | {'Improvement':<15s}")
print("-" * 80)
print(f"{'Linear Gaussian':<30s} | {score_linear:>15.2f} | {'baseline':<15s}")
print(f"{'Decision Trees (depth=2)':<30s} | {score_tree:>15.2f} | {score_tree - score_linear:>+15.2f}")
print(f"{'Mixed Theory':<30s} | {score_mixed:>15.2f} | {score_mixed - score_linear:>+15.2f}")

print()

best_approach = max([
    ('Linear Gaussian', score_linear),
    ('Decision Trees', score_tree),
    ('Mixed Theory', score_mixed)
], key=lambda x: x[1])

print(f"✓ Best approach: {best_approach[0]} (score: {best_approach[1]:.2f})")
print()

if score_mixed > score_linear * 1.05:
    print("✓ Mixed theory assignment provides significant improvement!")
    print("  Variable-specific theory languages work at larger scale (14 variables).")
elif score_tree > score_linear * 1.05:
    print("✓ Decision trees provide improvement")
    print("  Tree-based models capture relationships better for this dataset.")
else:
    print("≈ Linear Gaussian performs well")
    print("  Continuous relationships dominate in Wine dataset.")

print()
print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)
print()
print("Note: With 14 variables, we used beam search (not exhaustive enumeration).")
print("Results show approximate best structures, not guaranteed global optima.")
