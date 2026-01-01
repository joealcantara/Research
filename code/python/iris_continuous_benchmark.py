"""
Iris dataset benchmark with CONTINUOUS variables (Linear Gaussian Bayesian Networks).

Compares against the binary discretization benchmark to see if continuous models
better capture the structure in Iris data.

Uses:
- Continuous features: sepal/petal measurements (Gaussian + linear regression)
- Categorical target: species (1=setosa, 2=versicolor, 3=virginica)
- All 29,281 possible DAG structures
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from scripts.dag_utils import generate_all_dags, count_edges
from scripts.inference import compute_posteriors
from scripts.utils import print_posteriors, print_posterior_summary

print("=" * 80)
print("Iris Dataset - Continuous Variables Benchmark")
print("=" * 80)
print()

# Load Iris dataset (keep continuous features)
print("STEP 1: Load Iris dataset (continuous)")
print("-" * 80)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = iris.target + 1  # 1, 2, 3

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nSpecies distribution:")
print(df['species'].value_counts().sort_index())
print()

print("Variable types:")
for col in df.columns:
    if col == 'species':
        print(f"  {col}: categorical (1=setosa, 2=versicolor, 3=virginica)")
    else:
        print(f"  {col}: continuous (mean={df[col].mean():.2f}, std={df[col].std():.2f})")
print()

# Generate all possible DAG structures
print("STEP 2: Enumerate all possible DAG structures")
print("-" * 80)

variables = list(df.columns)
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
print("STEP 3: Compute posterior probabilities")
print("-" * 80)
print("Running Bayesian inference with λ = 2.0 (complexity penalty)")
print("Using Linear Gaussian models for continuous variables")
print()

results = compute_posteriors(df, all_dags, lambda_penalty=2.0)

print(f"Computed posteriors for all {len(results):,} structures")
print()

# Analyze results
print("STEP 4: Analyze results")
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
print("COMPARISON: Binary Discretization vs Continuous")
print("=" * 80)
print()
print("Binary discretization benchmark results (from iris_benchmark.py):")
print("  Top 1 posterior:        0.033151 (3.3%)")
print("  Top 10 cumulative:      0.234527 (23.5%)")
print("  Top 100 cumulative:     0.601573 (60.2%)")
print()
print("Continuous variables benchmark results (this run):")
print(f"  Top 1 posterior:        {top_result['posterior']:.6f} ({top_result['posterior']*100:.1f}%)")

top_10_prob = sum(r[1]['posterior'] for r in sorted_results[:10])
print(f"  Top 10 cumulative:      {top_10_prob:.6f} ({top_10_prob*100:.1f}%)")

top_100_prob = sum(r[1]['posterior'] for r in sorted_results[:100])
print(f"  Top 100 cumulative:     {top_100_prob:.6f} ({top_100_prob*100:.1f}%)")
print()

improvement_top1 = top_result['posterior'] / 0.033151
improvement_top10 = top_10_prob / 0.234527
improvement_top100 = top_100_prob / 0.601573

print("Improvement factor:")
print(f"  Top 1:    {improvement_top1:.1f}x")
print(f"  Top 10:   {improvement_top10:.1f}x")
print(f"  Top 100:  {improvement_top100:.1f}x")
print()

if improvement_top1 > 1.5:
    print("✓ CONTINUOUS variables provide BETTER structure identification!")
    print("  Linear Gaussian models better capture relationships than binary discretization.")
elif improvement_top1 < 0.7:
    print("✗ Binary discretization was better (unexpected!)")
else:
    print("≈ Similar performance (binary vs continuous)")

print()
print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)
