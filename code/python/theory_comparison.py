"""
Option 4: Greedy Theory Upgrade

Starting from the best Linear Gaussian structure, compare linear vs decision tree
models for each variable to see which benefits from logical rules.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from scripts.dag_utils import generate_all_dags, count_edges
from scripts.inference import compute_posteriors, _get_variable_type

print("=" * 80)
print("Theory Language Comparison: Linear vs Decision Trees")
print("=" * 80)
print()

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = iris.target + 1

print("Step 1: Find best Linear Gaussian structure")
print("-" * 80)

# Generate all DAGs
variables = list(df.columns)
all_dags = generate_all_dags(variables)

# Get best structure from Linear Gaussian
results_linear = compute_posteriors(df, all_dags, lambda_penalty=2.0)
sorted_results = sorted(results_linear.items(), key=lambda x: -x[1]['posterior'])
best_idx, best_result = sorted_results[0]
best_structure = all_dags[best_idx]

print(f"Best structure (Linear Gaussian): {best_result['posterior']:.4f} posterior")
print()
for var in variables:
    parents = best_structure[var]
    if len(parents) == 0:
        print(f"  {var}: (no parents)")
    else:
        print(f"  {var}: ← {', '.join(parents)}")
print()

print("Step 2: Compare linear vs tree for each variable")
print("-" * 80)
print()

# For each variable, compare linear vs tree models
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


# Compare for each variable
print(f"{'Variable':<15} | {'Linear':<12} | {'Tree (d=2)':<12} | {'Improvement':<12} | {'Winner':<10}")
print("-" * 80)

improvements = {}
for var in variables:
    parents = best_structure[var]

    if len(parents) == 0:
        print(f"{var:<15} | (no parents - same for both)")
        continue

    linear_score = score_variable_with_theory(df, best_structure, var, 'linear')
    tree_score = score_variable_with_theory(df, best_structure, var, 'tree', max_depth=2)

    improvement = tree_score - linear_score
    improvements[var] = improvement

    winner = "Tree" if improvement > 1.0 else "Linear" if improvement < -1.0 else "Tie"

    print(f"{var:<15} | {linear_score:>12.2f} | {tree_score:>12.2f} | {improvement:>+12.2f} | {winner:<10}")

print()
print("=" * 80)
print("Summary")
print("=" * 80)
print()

# Total improvement
total_improvement = sum(improvements.values())
print(f"Total improvement from upgrading all to trees: {total_improvement:+.2f}")
print()

# Which variables benefit most
sorted_improvements = sorted(improvements.items(), key=lambda x: -x[1])
print("Variables that benefit most from decision trees:")
for var, imp in sorted_improvements[:3]:
    if imp > 1.0:
        print(f"  {var}: {imp:+.2f} (logical rules help!)")
    else:
        print(f"  {var}: {imp:+.2f} (linear is fine)")
print()

print("Variables that prefer linear models:")
for var, imp in sorted_improvements[-3:]:
    if imp < -1.0:
        print(f"  {var}: {imp:+.2f} (linear is better)")
    else:
        print(f"  {var}: {imp:+.2f} (either works)")
print()

# Estimate posterior with mixed theories
print("=" * 80)
print("Recommendation")
print("=" * 80)
print()

if total_improvement > 10:
    print("✓ Decision trees provide significant improvement!")
    print(f"  Upgrading all variables: ~{total_improvement:.1f} log-likelihood improvement")
    print()
    print("Suggested mixed theory assignment:")
    for var in variables:
        if var in improvements:
            theory = "Decision Tree" if improvements[var] > 1.0 else "Linear"
            print(f"  {var}: {theory}")
elif total_improvement < -10:
    print("✓ Linear models are better overall")
    print("  Stick with Linear Gaussian Bayesian Networks")
else:
    print("≈ Mixed results - both theories have merit")
    print("  Consider using both depending on the variable")
