"""
Option 1: Mixed Theory Assignment

Use the best structure from Linear Gaussian, but assign variable-specific theories:
- Decision trees for variables that benefit from logical rules (species, petal_width)
- Linear regression for variables that prefer continuous relationships (petal_length, sepal_length)

This should outperform both pure Linear Gaussian (17.5%) and uniform decision trees.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from scripts.dag_utils import generate_all_dags, count_edges
from scripts.inference import compute_posteriors, _get_variable_type

print("=" * 80)
print("Mixed Theory Assignment Benchmark")
print("=" * 80)
print()

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = iris.target + 1

print("Step 1: Find best structure using Linear Gaussian")
print("-" * 80)

# Generate all DAGs
variables = list(df.columns)
all_dags = generate_all_dags(variables)

# Get best structure from Linear Gaussian
results_linear = compute_posteriors(df, all_dags, lambda_penalty=2.0)
sorted_results = sorted(results_linear.items(), key=lambda x: -x[1]['posterior'])
best_idx, best_result = sorted_results[0]
best_structure = all_dags[best_idx]

print(f"Best structure: {best_result['posterior']:.4f} posterior ({best_result['posterior']*100:.1f}%)")
print()
for var in variables:
    parents = best_structure[var]
    if len(parents) == 0:
        print(f"  {var}: (no parents)")
    else:
        print(f"  {var}: ← {', '.join(parents)}")
print()


def score_variable_with_theory(df, structure, var, theory_type='linear', max_depth=2):
    """
    Score a single variable using specified theory type.

    Returns log-likelihood contribution for this variable only.
    """
    parents = structure[var]
    var_type = _get_variable_type(df, var)

    if len(parents) == 0:
        # No parents - both theories give same result (marginal distribution)
        if var_type == 'continuous':
            mean = df[var].mean()
            std = df[var].std() + 1e-6
            log_lik = -0.5 * np.sum(np.log(2 * np.pi * std**2) + ((df[var] - mean) / std)**2)
        else:
            # Categorical marginal
            unique_vals = sorted(df[var].unique())
            total = len(df)
            num_values = len(unique_vals)
            log_lik = 0.0
            for val in unique_vals:
                count = (df[var] == val).sum()
                prob = (count + 1) / (total + num_values)
                log_lik += (df[var] == val).sum() * np.log(prob)

        return log_lik

    # Has parents
    if theory_type == 'linear':
        # Linear regression (for continuous) or CPT (for categorical)
        if var_type == 'continuous':
            X = df[parents].values
            y = df[var].values

            # Linear regression
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
            # Categorical with continuous parents - discretize at median
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

            log_lik = 0.0
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
            tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            tree.fit(X, y)
            predictions = tree.predict(X)
            residuals = y - predictions
            residual_std = np.std(residuals) + 1e-6

            log_lik = -0.5 * np.sum(np.log(2 * np.pi * residual_std**2) + (residuals / residual_std)**2)
        else:
            # Classification tree
            tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            tree.fit(X, y)

            log_lik = 0.0
            for idx in range(len(df)):
                probs = tree.predict_proba(X[idx:idx+1])[0]
                class_idx = int(y[idx] - min(y))  # Adjust for 1-indexing
                prob = probs[class_idx] if class_idx < len(probs) else 1e-10
                log_lik += np.log(prob + 1e-10)

    return log_lik


def compute_mixed_theory_score(df, structure, theory_assignment, lambda_penalty=2.0):
    """
    Compute log-likelihood and posterior for a structure with mixed theory assignment.

    Args:
        df: DataFrame
        structure: DAG structure
        theory_assignment: Dict mapping variables to theory types ('linear' or 'tree')
        lambda_penalty: Complexity penalty per edge

    Returns:
        Dict with log_likelihood, edges, log_prior, log_posterior, posterior
    """
    # Score each variable with its assigned theory
    log_lik = 0.0
    for var in structure.keys():
        theory_type = theory_assignment.get(var, 'linear')
        var_score = score_variable_with_theory(df, structure, var, theory_type, max_depth=2)
        log_lik += var_score

    # Count edges for complexity penalty
    edges = sum(len(parents) for parents in structure.values())

    # Compute posterior
    log_prior = -lambda_penalty * edges
    log_posterior = log_lik + log_prior

    return {
        'log_likelihood': log_lik,
        'edges': edges,
        'log_prior': log_prior,
        'log_posterior': log_posterior
    }


print("Step 2: Define theory assignment strategy")
print("-" * 80)
print()

# Mixed Theory Assignment (based on theory_comparison.py findings)
print("Mixed Theory Assignment Strategy:")
print("  Species → Decision Tree (logical rules for classification)")
print("  Petal_width → Decision Tree (marginal benefit)")
print("  Petal_length → Linear (much better for continuous relationships)")
print("  Sepal_length → Linear (better for continuous relationships)")
print("  Sepal_width → Linear (default)")
print()

mixed_theory_assignment = {
    'species': 'tree',       # +14.23 improvement
    'petal_width': 'tree',   # +2.61 improvement
    'petal_length': 'linear', # -52.57 with trees (linear much better)
    'sepal_length': 'linear', # -5.56 with trees (linear better)
    'sepal_width': 'linear'   # Default to linear (no parents)
}

print("Step 3: Score all structures with mixed theory assignment")
print("-" * 80)
print("Computing posteriors for all 29,281 structures...")
print("(This may take a few minutes)")
print()

# Compute posteriors for all structures using mixed theory
mixed_results = {}
for idx, structure in enumerate(all_dags):
    if (idx + 1) % 5000 == 0:
        print(f"  Progress: {idx + 1:,} / {len(all_dags):,} structures")

    result = compute_mixed_theory_score(df, structure, mixed_theory_assignment, lambda_penalty=2.0)
    mixed_results[idx] = result

# Normalize posteriors
log_posteriors = np.array([r['log_posterior'] for r in mixed_results.values()])
log_posteriors_shifted = log_posteriors - log_posteriors.max()
posteriors = np.exp(log_posteriors_shifted)
posteriors = posteriors / posteriors.sum()

for i, idx in enumerate(mixed_results.keys()):
    mixed_results[idx]['posterior'] = posteriors[i]

print(f"\nComputed posteriors for all {len(mixed_results):,} structures")
print()

print("Step 4: Analyze posterior distribution")
print("-" * 80)
print()

# Sort by posterior
sorted_mixed = sorted(mixed_results.items(), key=lambda x: -x[1]['posterior'])

# Get top structure
top_idx, top_result = sorted_mixed[0]
top_structure = all_dags[top_idx]

print("Top structure (Mixed Theory):")
for var in variables:
    parents = top_structure[var]
    if len(parents) == 0:
        print(f"  {var}: (no parents)")
    else:
        print(f"  {var}: ← {', '.join(parents)}")
print()

# Compute cumulative posteriors
top_1_prob = top_result['posterior']
top_10_prob = sum(r[1]['posterior'] for r in sorted_mixed[:10])
top_100_prob = sum(r[1]['posterior'] for r in sorted_mixed[:100])

print("=" * 80)
print("COMPARISON: Binary vs Linear Gaussian vs Mixed Theory")
print("=" * 80)
print()

print("Binary discretization (from iris_benchmark.py):")
print("  Top 1 posterior:        3.3%")
print("  Top 10 cumulative:     23.5%")
print("  Top 100 cumulative:    60.2%")
print()

print("Linear Gaussian (from iris_continuous_benchmark.py):")
print(f"  Top 1 posterior:       {best_result['posterior']*100:.1f}%")
print("  Top 10 cumulative:     76.3%")
print("  Top 100 cumulative:   100.0%")
print()

print("Mixed Theory Assignment (this run):")
print(f"  Top 1 posterior:       {top_1_prob*100:.1f}%")
print(f"  Top 10 cumulative:     {top_10_prob*100:.1f}%")
print(f"  Top 100 cumulative:    {top_100_prob*100:.1f}%")
print()

# Improvement factors
improvement_top1 = top_1_prob / best_result['posterior']
print("Improvement over Linear Gaussian:")
print(f"  Top 1:    {improvement_top1:.2f}x")
print()

if top_1_prob > best_result['posterior'] * 1.1:
    print("✓ Mixed theory assignment provides better structure identification!")
    print("  Using variable-specific theory languages beats uniform approaches.")
elif top_1_prob < best_result['posterior'] * 0.9:
    print("✗ Mixed theory performed worse (unexpected!)")
else:
    print("≈ Similar performance to Linear Gaussian")
    print("  Structure matters more than theory assignment for this dataset.")

print()
print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)
