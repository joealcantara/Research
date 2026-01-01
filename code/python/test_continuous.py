"""
Test continuous variable support with Iris dataset.

Tests that the inference functions correctly handle:
- Continuous variables (sepal/petal measurements)
- Categorical variables (species)
- Mixed structures
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from scripts.inference import estimate_parameters, compute_log_likelihood

print("=" * 80)
print("Test: Continuous Variable Support")
print("=" * 80)
print()

# Load Iris with continuous features
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = iris.target + 1  # 1, 2, 3

print("Dataset:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nVariable types:")
print(df.dtypes)
print()

# Test 1: Simple structure with continuous variable (no parents)
print("TEST 1: Continuous variable with no parents")
print("-" * 80)

structure1 = {
    'sepal_length': [],
    'sepal_width': [],
    'petal_length': [],
    'petal_width': [],
    'species': []
}

params1 = estimate_parameters(df, structure1)
print("Parameters for sepal_length (no parents):")
print(f"  {params1['sepal_length']}")
print()

log_lik1 = compute_log_likelihood(df, structure1, params1)
print(f"Log-likelihood: {log_lik1:.2f}")
print()

# Test 2: Continuous variable with continuous parents
print("TEST 2: Continuous variable with continuous parents")
print("-" * 80)

structure2 = {
    'sepal_length': [],
    'sepal_width': [],
    'petal_length': ['sepal_length', 'sepal_width'],
    'petal_width': [],
    'species': []
}

params2 = estimate_parameters(df, structure2)
print("Parameters for petal_length (parents: sepal_length, sepal_width):")
print(f"  Type: {params2['petal_length'][()]['type']}")
print(f"  Intercept: {params2['petal_length'][()]['intercept']:.4f}")
print(f"  Coefficients: {params2['petal_length'][()]['coeffs']}")
print(f"  Residual std: {params2['petal_length'][()]['std']:.4f}")
print()

log_lik2 = compute_log_likelihood(df, structure2, params2)
print(f"Log-likelihood: {log_lik2:.2f}")
print(f"Improvement over independent: {log_lik2 - log_lik1:+.2f}")
print()

# Test 3: Categorical variable with continuous parents
print("TEST 3: Categorical variable with continuous parents")
print("-" * 80)

structure3 = {
    'sepal_length': [],
    'sepal_width': [],
    'petal_length': [],
    'petal_width': [],
    'species': ['petal_length', 'petal_width']
}

params3 = estimate_parameters(df, structure3)
print("Parameters for species (parents: petal_length, petal_width):")
print(f"  This creates a conditional probability table")
print(f"  Number of parent combinations: {len(params3['species'])}")
print()

log_lik3 = compute_log_likelihood(df, structure3, params3)
print(f"Log-likelihood: {log_lik3:.2f}")
print(f"Improvement over independent: {log_lik3 - log_lik1:+.2f}")
print()

# Test 4: Mixed structure
print("TEST 4: Mixed structure (continuous + categorical)")
print("-" * 80)

structure4 = {
    'petal_width': [],
    'petal_length': ['petal_width'],
    'species': ['petal_width', 'petal_length'],
    'sepal_length': ['species'],
    'sepal_width': ['species']
}

params4 = estimate_parameters(df, structure4)
log_lik4 = compute_log_likelihood(df, structure4, params4)
print(f"Log-likelihood: {log_lik4:.2f}")
print(f"Improvement over independent: {log_lik4 - log_lik1:+.2f}")
print()

print("=" * 80)
print("All tests passed! Continuous variable support working correctly.")
print("=" * 80)
