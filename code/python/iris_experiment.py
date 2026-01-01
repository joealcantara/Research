"""
Scientist AI: Gaussian Linear Bayesian Networks on Iris Dataset

This script demonstrates Bayesian causal discovery using continuous data
with linear regression models.
"""

import numpy as np
import pandas as pd
from itertools import combinations, product
import time
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression

print("="*80)
print("SCIENTIST AI: GAUSSIAN LINEAR BAYESIAN NETWORKS")
print("="*80)
print()

# ============================================================================
# 1. LOAD IRIS DATASET
# ============================================================================
print("\n" + "-"*80)
print("1. LOAD IRIS DATASET")
print("-"*80)
print()

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['species'] = iris.target

# Rename columns for easier handling
df_continuous = df_iris.copy()
df_continuous.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

print("Dataset:")
print(df_continuous.head())
print(f"\nShape: {df_continuous.shape}")
print(f"\nVariable types:")
print("  - sepal_length, sepal_width, petal_length, petal_width: continuous")
print("  - species: categorical (0=setosa, 1=versicolor, 2=virginica)")


# ============================================================================
# 2. DAG GENERATION FUNCTIONS
# ============================================================================
print("\n" + "-"*80)
print("2. DAG GENERATION FUNCTIONS")
print("-"*80)
print()


def count_edges(structure):
    """Count total edges in a structure."""
    return sum(len(parents) for parents in structure.values())


def is_dag(edges, n):
    """Check if a set of directed edges forms a DAG (no cycles)."""
    adj = {i: [] for i in range(n)}
    for i, j in edges:
        adj[i].append(j)

    visited = [False] * n
    rec_stack = [False] * n

    def has_cycle(node):
        visited[node] = True
        rec_stack[node] = True

        for neighbor in adj[node]:
            if not visited[neighbor]:
                if has_cycle(neighbor):
                    return True
            elif rec_stack[neighbor]:
                return True

        rec_stack[node] = False
        return False

    for i in range(n):
        if not visited[i]:
            if has_cycle(i):
                return False

    return True


def generate_all_dags(variables):
    """Generate all possible DAG structures for given variables."""
    n = len(variables)
    all_edges = [(i, j) for i in range(n) for j in range(n) if i != j]

    print(f"Generating DAGs for {n} variables...")
    print(f"Possible edges: {len(all_edges)}")
    print(f"Edge subsets to check: {2**len(all_edges):,}")

    dags = []
    start_time = time.time()

    for num_edges in range(len(all_edges) + 1):
        for edge_subset in combinations(all_edges, num_edges):
            if is_dag(edge_subset, n):
                structure = {}
                for i, var in enumerate(variables):
                    parents = [variables[j] for j, k in edge_subset if k == i]
                    structure[var] = parents
                dags.append(structure)

    elapsed = time.time() - start_time
    print(f"Generated {len(dags):,} DAGs in {elapsed:.2f} seconds")

    return dags


# ============================================================================
# 3. GAUSSIAN LINEAR BN FUNCTIONS
# ============================================================================
print("\n" + "-"*80)
print("3. GAUSSIAN LINEAR BN FUNCTIONS")
print("-"*80)
print()


def estimate_parameters_gaussian(df, structure):
    """
    Fit linear/logistic regression models for Gaussian BN.
    - Continuous variables: linear regression
    - Categorical variable (species): logistic regression
    """
    models = {}

    for var, parents in structure.items():
        if len(parents) == 0:
            # No parents: store marginal statistics
            if var == 'species':
                models[var] = {
                    'type': 'categorical_marginal',
                    'probs': df[var].value_counts(normalize=True).sort_index().values
                }
            else:
                models[var] = {
                    'type': 'continuous_marginal',
                    'mean': df[var].mean(),
                    'std': df[var].std()
                }
        else:
            # Has parents: fit regression model
            X = df[parents].values
            y = df[var].values

            if var == 'species':
                # Categorical: logistic regression
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X, y)
                models[var] = {
                    'type': 'categorical_regression',
                    'model': model,
                    'parents': parents
                }
            else:
                # Continuous: linear regression
                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X)
                residuals = y - predictions
                models[var] = {
                    'type': 'continuous_regression',
                    'model': model,
                    'parents': parents,
                    'residual_std': np.std(residuals) + 1e-6
                }

    return models


def compute_log_likelihood_gaussian(df, structure, models):
    """Compute log probability using Gaussian BN (vectorized)."""
    log_lik = 0.0

    for var in structure.keys():
        model = models[var]

        if model['type'] == 'categorical_marginal':
            # Marginal categorical
            for class_idx in range(len(model['probs'])):
                count = (df[var] == class_idx).sum()
                if count > 0:
                    log_lik += count * np.log(model['probs'][class_idx] + 1e-10)

        elif model['type'] == 'continuous_marginal':
            # Marginal continuous (vectorized)
            values = df[var].values
            mean = model['mean']
            std = model['std']
            log_lik += np.sum(-0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((values - mean) / std)**2)

        elif model['type'] == 'categorical_regression':
            # Conditional categorical (vectorized)
            X = df[model['parents']].values
            probs = model['model'].predict_proba(X)
            y = df[var].values.astype(int)
            for i, class_idx in enumerate(y):
                log_lik += np.log(probs[i, class_idx] + 1e-10)

        elif model['type'] == 'continuous_regression':
            # Conditional continuous (vectorized)
            X = df[model['parents']].values
            predictions = model['model'].predict(X)
            values = df[var].values
            std = model['residual_std']
            log_lik += np.sum(-0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((values - predictions) / std)**2)

    return log_lik


def compute_posteriors_gaussian(df, structures, lambda_penalty=2.0):
    """Compute posterior probability using Gaussian linear BN."""
    print(f"Computing posteriors with Gaussian linear BN for {len(structures):,} structures...")
    print(f"  lambda={lambda_penalty}")
    start_time = time.time()

    results = []

    for i, structure in enumerate(structures):
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1:,} / {len(structures):,} ({elapsed:.1f}s elapsed)")

        models = estimate_parameters_gaussian(df, structure)
        log_lik = compute_log_likelihood_gaussian(df, structure, models)
        edges = count_edges(structure)

        log_prior = -lambda_penalty * edges
        log_posterior = log_lik + log_prior

        results.append({
            'structure': structure,
            'log_likelihood': log_lik,
            'edges': edges,
            'log_prior': log_prior,
            'log_posterior': log_posterior,
            'models': models
        })

    # Normalize to get probabilities
    log_posteriors = np.array([r['log_posterior'] for r in results])
    log_posteriors_shifted = log_posteriors - log_posteriors.max()
    posteriors = np.exp(log_posteriors_shifted)
    posteriors = posteriors / posteriors.sum()

    for i in range(len(results)):
        results[i]['posterior'] = posteriors[i]

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    return results


# ============================================================================
# 4. GENERATE DAGS
# ============================================================================
print("\n" + "-"*80)
print("4. GENERATE ALL DAGS FOR 5 VARIABLES")
print("-"*80)
print()

variables = list(df_continuous.columns)
print(f"Variables: {variables}")
print()

all_dags = generate_all_dags(variables)


# ============================================================================
# 5. RUN GAUSSIAN LINEAR BN INFERENCE
# ============================================================================
print("\n" + "-"*80)
print("5. RUN GAUSSIAN LINEAR BN INFERENCE")
print("-"*80)
print()

results_gaussian = compute_posteriors_gaussian(df_continuous, all_dags, lambda_penalty=2.0)


# ============================================================================
# 6. ANALYZE RESULTS
# ============================================================================
print("\n" + "-"*80)
print("6. ANALYZE RESULTS")
print("-"*80)
print()

sorted_results = sorted(results_gaussian, key=lambda x: -x['posterior'])

print("Top 20 structures by posterior probability:")
print(f"{'Rank':4} | {'Edges':5} | {'Log-lik':>10} | {'Log-post':>10} | {'Posterior':>10} | Structure")
print("-" * 100)

for rank, r in enumerate(sorted_results[:20], 1):
    edges_list = []
    for var, parents in r['structure'].items():
        for parent in parents:
            edges_list.append(f"{parent}→{var}")

    if len(edges_list) == 0:
        structure_str = "(independent)"
    else:
        structure_str = ", ".join(edges_list[:3])
        if len(edges_list) > 3:
            structure_str += f" (+{len(edges_list)-3} more)"

    print(f"{rank:4} | {r['edges']:5} | {r['log_likelihood']:10.2f} | {r['log_posterior']:10.2f} | {r['posterior']:10.6f} | {structure_str}")

print(f"\nPosterior concentration:")
print(f"  Top 1:   {sorted_results[0]['posterior']:.6f}")
print(f"  Top 5:   {sum(r['posterior'] for r in sorted_results[:5]):.6f}")
print(f"  Top 10:  {sum(r['posterior'] for r in sorted_results[:10]):.6f}")
print(f"  Top 20:  {sum(r['posterior'] for r in sorted_results[:20]):.6f}")
print(f"  Top 100: {sum(r['posterior'] for r in sorted_results[:100]):.6f}")

print("\n" + "="*80)
print("MAP (Maximum A Posteriori) Structure:")
print("="*80)

top_structure = sorted_results[0]['structure']
for var in variables:
    parents = top_structure[var]
    if len(parents) == 0:
        print(f"{var}: (no parents)")
    else:
        print(f"{var}: ← {', '.join(parents)}")

print(f"\nTotal edges: {count_edges(top_structure)}")
print(f"Posterior probability: {sorted_results[0]['posterior']:.6f}")
print()
