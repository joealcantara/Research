#!/usr/bin/env python3
"""
Exhaustive search for 3-variable chain using Python reference implementation.

This script enumerates all 25 possible DAGs and computes the posterior
distribution using BDeu scoring to compare with Rust implementation.
"""

import numpy as np
import pandas as pd
import json
from itertools import combinations, product
from collections import defaultdict
from scipy.special import gammaln

def is_dag(adj_matrix):
    """Check if adjacency matrix represents a DAG (no cycles)."""
    n = len(adj_matrix)
    # Use topological sort approach
    visited = set()
    rec_stack = set()

    def has_cycle(node):
        visited.add(node)
        rec_stack.add(node)

        for neighbor in range(n):
            if adj_matrix[node][neighbor]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

        rec_stack.remove(node)
        return False

    for node in range(n):
        if node not in visited:
            if has_cycle(node):
                return False
    return True

def enumerate_all_dags(n_vars):
    """Enumerate all possible DAGs for n variables."""
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n_vars) for j in range(n_vars) if i != j]
    n_possible_edges = len(all_edges)

    dags = []

    # Enumerate all subsets of edges
    for subset_bits in range(2 ** n_possible_edges):
        # Create adjacency matrix for this subset
        adj_matrix = [[False] * n_vars for _ in range(n_vars)]

        for edge_idx, (i, j) in enumerate(all_edges):
            if (subset_bits >> edge_idx) & 1:
                adj_matrix[i][j] = True

        # Check if it's a DAG
        if is_dag(adj_matrix):
            dags.append(adj_matrix)

    return dags

def bdeu_score(data, structure, alpha=1.0):
    """
    Compute BDeu score for a structure.

    Args:
        data: DataFrame with binary variables
        structure: Adjacency matrix (parent[i][j] = True means i -> j)
        alpha: BDeu hyperparameter

    Returns:
        Log marginal likelihood
    """
    n_vars = len(structure)
    score = 0.0

    for j in range(n_vars):
        # Find parents of variable j
        parents = [i for i in range(n_vars) if structure[i][j]]

        # Get data for this variable and its parents
        var_col = data.iloc[:, j].values

        if len(parents) == 0:
            # No parents - just marginal distribution
            n = len(var_col)
            n1 = np.sum(var_col == 1)
            n0 = n - n1

            # BDeu score with Dirichlet prior
            r = 2  # binary variable
            alpha_ijk = alpha / r

            score += (gammaln(alpha) - gammaln(alpha + n) +
                     gammaln(alpha_ijk + n0) + gammaln(alpha_ijk + n1) -
                     2 * gammaln(alpha_ijk))
        else:
            # Has parents - conditional distribution
            parent_data = data.iloc[:, parents].values

            # Count occurrences for each parent configuration
            parent_configs = defaultdict(lambda: {'total': 0, 'n0': 0, 'n1': 0})

            for row_idx in range(len(var_col)):
                parent_vals = tuple(parent_data[row_idx])
                child_val = var_col[row_idx]

                parent_configs[parent_vals]['total'] += 1
                if child_val == 0:
                    parent_configs[parent_vals]['n0'] += 1
                else:
                    parent_configs[parent_vals]['n1'] += 1

            # BDeu score
            r = 2  # binary variable
            q = 2 ** len(parents)  # number of parent configurations
            alpha_ij = alpha / q
            alpha_ijk = alpha_ij / r

            for parent_config, counts in parent_configs.items():
                n_ij = counts['total']
                n0 = counts['n0']
                n1 = counts['n1']

                score += (gammaln(alpha_ij) - gammaln(alpha_ij + n_ij) +
                         gammaln(alpha_ijk + n0) + gammaln(alpha_ijk + n1) -
                         2 * gammaln(alpha_ijk))

    return score

def main():
    # Load data
    data_path = '../data/chain_3var.csv'
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)

    n_samples = len(data)
    n_vars = len(data.columns)
    print(f"Loaded {n_samples} samples, {n_vars} variables")
    print(f"Variables: {list(data.columns)}")

    # Enumerate all DAGs
    print(f"\nEnumerating all possible DAG structures...")
    all_dags = enumerate_all_dags(n_vars)
    print(f"Found {len(all_dags)} DAGs")

    # Score all structures
    print(f"\nScoring all structures with BDeu...")
    scores = []

    for dag in all_dags:
        # Create name from edges
        edges = []
        for i in range(n_vars):
            for j in range(n_vars):
                if dag[i][j]:
                    edges.append(f"{data.columns[i]}->{data.columns[j]}")

        name = ",".join(sorted(edges)) if edges else "empty"

        # Compute score
        log_score = bdeu_score(data, dag, alpha=1.0)
        edge_count = sum(sum(row) for row in dag)

        scores.append({
            'name': name,
            'edges': edge_count,
            'log_likelihood': log_score,  # BDeu is marginal likelihood
            'log_posterior': log_score,   # Uniform prior
            'structure': dag
        })

    # Normalize to get posterior probabilities
    log_scores = np.array([s['log_posterior'] for s in scores])
    max_log_score = np.max(log_scores)
    log_posteriors = log_scores - max_log_score
    posteriors = np.exp(log_posteriors)
    posteriors /= np.sum(posteriors)

    for i, score in enumerate(scores):
        score['posterior'] = posteriors[i]

    # Sort by posterior
    scores.sort(key=lambda x: x['posterior'], reverse=True)

    # Print top 10
    print("\n=== Top 10 Structures ===")
    for i, score in enumerate(scores[:10]):
        print(f"{i+1:2}. {score['name']:30} (edges: {score['edges']}, "
              f"posterior: {score['posterior']:.6f}, log_post: {score['log_posterior']:.3f})")

    # Save results
    output_path = '../results/exhaustive_python.json'
    print(f"\nSaving results to: {output_path}")

    results = {
        'n_samples': n_samples,
        'n_variables': n_vars,
        'n_structures': len(scores),
        'scoring_method': 'BDeu',
        'structures': [
            {
                'name': s['name'],
                'edges': s['edges'],
                'log_likelihood': s['log_likelihood'],
                'log_prior': 0.0,  # Uniform
                'log_posterior': s['log_posterior'],
                'posterior': s['posterior'],
            }
            for s in scores
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary stats
    print("\n=== Summary Statistics ===")
    print(f"Total probability mass: {sum(s['posterior'] for s in scores):.6f}")
    entropy = -sum(s['posterior'] * np.log(s['posterior'])
                   for s in scores if s['posterior'] > 0)
    print(f"Entropy: {entropy:.3f} nats")

    print("\nDone!")

if __name__ == '__main__':
    main()
