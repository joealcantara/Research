"""
Iris dataset benchmark for Bayesian causal structure learning.

Tests structure learning on the Iris dataset:
- 4 continuous features (sepal length, sepal width, petal length, petal width)
- 1 categorical target (species: 1=setosa, 2=versicolor, 3=virginica)
- 5 variables total → 29,281 possible DAG structures

Strategy:
- Discretize continuous features to binary (above/below median)
- Keep species as 1, 2, 3
- Enumerate all DAGs
- Compute posteriors
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from scripts.dag_utils import generate_all_dags, count_edges
from scripts.inference import compute_posteriors
from scripts.utils import print_posteriors, print_posterior_summary

print("=" * 80)
print("Iris Dataset - Bayesian Causal Structure Learning Benchmark")
print("=" * 80)
print()

# Load Iris dataset
print("STEP 1: Load Iris dataset")
print("-" * 80)
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['species'] = iris.target + 1  # Convert to 1, 2, 3 (instead of 0, 1, 2)

print(f"Dataset shape: {df_iris.shape}")
print(f"\nFirst few rows:")
print(df_iris.head())
print(f"\nSpecies distribution:")
print(df_iris['species'].value_counts().sort_index())
print()

# Discretize continuous features to binary
print("STEP 2: Discretize continuous features to binary")
print("-" * 80)

df_binary = pd.DataFrame()

# Continuous features: 1 if above median, 0 otherwise
for col in ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']:
    median = df_iris[col].median()
    short_name = col.split()[0]  # 'sepal' or 'petal'
    dimension = col.split()[1]   # 'length' or 'width'
    df_binary[f'{short_name}_{dimension}'] = (df_iris[col] > median).astype(int)

# Species: keep as 1, 2, 3
df_binary['species'] = df_iris['species']

print("Discretized dataset:")
print(df_binary.head(10))
print(f"\nShape: {df_binary.shape}")
print(f"\nVariable names: {list(df_binary.columns)}")
print(f"\nVariable types:")
for col in df_binary.columns:
    if col == 'species':
        print(f"  {col}: categorical (1=setosa, 2=versicolor, 3=virginica)")
    else:
        print(f"  {col}: binary (0/1)")
print()

# Generate all possible DAG structures
print("STEP 3: Enumerate all possible DAG structures")
print("-" * 80)

variables = list(df_binary.columns)
all_dags = generate_all_dags(variables)

print(f"Generated {len(all_dags):,} possible DAG structures for {len(variables)} variables")
print()

# Show edge distribution
edge_counts = {}
for dag in all_dags:
    edges = count_edges(dag)
    edge_counts[edges] = edge_counts.get(edges, 0) + 1

print("Edge distribution:")
for edges in sorted(edge_counts.keys()):
    print(f"  {edges} edges: {edge_counts[edges]:,} DAGs")
print()

# Compute posteriors
print("STEP 4: Compute posterior probabilities")
print("-" * 80)
print("Running Bayesian inference with λ = 2.0 (complexity penalty)")
print()

results = compute_posteriors(df_binary, all_dags, lambda_penalty=2.0)

print(f"Computed posteriors for all {len(results):,} structures")
print()

# Analyze results
print("STEP 5: Analyze results")
print("-" * 80)
print()

# Show top 20 structures
print("Top 20 structures by posterior probability:")
print()
print_posteriors(results, top_n=20)
print()

# Show posterior concentration
print_posterior_summary(results)
print()

# Show MAP structure
print("=" * 80)
print("MAP (Maximum A Posteriori) Structure")
print("=" * 80)
print()

sorted_results = sorted(results.items(), key=lambda x: -x[1]['posterior'])
top_idx, top_result = sorted_results[0]
top_structure = all_dags[top_idx]

for var in variables:
    parents = top_structure[var]
    if len(parents) == 0:
        print(f"{var}: (no parents - root node)")
    else:
        print(f"{var}: ← {', '.join(parents)}")

print(f"\nTotal edges: {count_edges(top_structure)}")
print(f"Posterior probability: {top_result['posterior']:.6f}")
print()

print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)
