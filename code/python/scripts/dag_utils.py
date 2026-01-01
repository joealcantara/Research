"""
DAG structure utilities for Bayesian causal discovery.

Functions for generating and working with directed acyclic graphs (DAGs).
"""

from itertools import combinations


def count_edges(structure):
    """Count total edges in a DAG structure."""
    return sum(len(parents) for parents in structure.values())


def is_dag(edges, n):
    """
    Check if a set of directed edges forms a DAG (no cycles).

    Uses depth-first search (DFS) to detect cycles.

    Args:
        edges: List of tuples (i, j) representing directed edges
        n: Number of nodes

    Returns:
        True if edges form a DAG (acyclic), False if there's a cycle
    """
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for i, j in edges:
        adj[i].append(j)

    # DFS with visited and recursion stack
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
                # Back edge found - cycle detected
                return True

        rec_stack[node] = False
        return False

    # Check each component
    for i in range(n):
        if not visited[i]:
            if has_cycle(i):
                return False

    return True


def generate_all_dags(variables):
    """
    Generate all possible DAG structures for a given set of variables.

    For n variables, we consider all possible directed edges between them.
    For each subset of edges, we check if it forms a DAG (acyclic).

    Examples:
        - 3 variables: 6 possible edges, 2^6 = 64 subsets, 25 valid DAGs
        - 5 variables: 20 possible edges, 2^20 = 1,048,576 subsets, 29,281 valid DAGs

    Args:
        variables: List of variable names (e.g., ['X1', 'X2', 'X3'])

    Returns:
        List of DAG structures, where each structure is a dict:
        {variable: [list of parent variables]}
    """
    n = len(variables)

    # All possible directed edges (i, j) where i != j
    all_edges = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Generate all subsets of edges
    dags = []

    for num_edges in range(len(all_edges) + 1):
        for edge_subset in combinations(all_edges, num_edges):
            # Check if this edge set forms a DAG (acyclic)
            if is_dag(edge_subset, n):
                # Convert to structure format: {variable: [list of parents]}
                structure = {}
                for i, var in enumerate(variables):
                    # Find parents of this variable (edges that point TO this variable)
                    parents = [variables[j] for j, k in edge_subset if k == i]
                    structure[var] = parents

                dags.append(structure)

    return dags
