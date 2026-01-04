#!/usr/bin/env python3
"""
Generate synthetic datasets with known DAG structures for testing.

Creates 10 diverse datasets with different causal structures to ensure
implementations handle various graph topologies correctly.
"""

import numpy as np
import json
from pathlib import Path

np.random.seed(42)  # Reproducible

def generate_linear_gaussian_data(structure, n_samples=2000, noise_std=1.0):
    """
    Generate data from a linear Gaussian DAG.

    Args:
        structure: Dict mapping node -> list of parents
        n_samples: Number of samples to generate
        noise_std: Standard deviation of Gaussian noise

    Returns:
        data: numpy array of shape (n_samples, n_vars)
    """
    n_vars = len(structure)
    data = np.zeros((n_samples, n_vars))

    # Topological sort to ensure parents are generated before children
    visited = set()
    topo_order = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for parent in structure[node]:
            dfs(parent)
        topo_order.append(node)

    for node in range(n_vars):
        dfs(node)

    # Generate data in topological order
    for node in topo_order:
        parents = structure[node]
        if not parents:
            # Root node: sample from standard normal
            data[:, node] = np.random.randn(n_samples) * noise_std
        else:
            # Child node: linear combination of parents + noise
            coefficients = np.random.uniform(0.5, 2.0, len(parents))
            # Randomly flip some signs
            coefficients *= np.random.choice([-1, 1], len(parents))

            data[:, node] = np.sum([coefficients[i] * data[:, p]
                                   for i, p in enumerate(parents)], axis=0)
            data[:, node] += np.random.randn(n_samples) * noise_std

    return data

def save_dataset(name, structure, data, description):
    """Save synthetic dataset and its metadata."""
    output_dir = Path("synthetic_datasets")
    output_dir.mkdir(exist_ok=True)

    # Save data
    np.savetxt(output_dir / f"{name}.csv", data, delimiter=',', fmt='%.6f')

    # Save metadata
    metadata = {
        "name": name,
        "description": description,
        "n_samples": data.shape[0],
        "n_vars": data.shape[1],
        "true_structure": {str(k): v for k, v in structure.items()},
        "edges": sum(len(parents) for parents in structure.values()),
    }

    with open(output_dir / f"{name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {name}: {data.shape[0]} samples × {data.shape[1]} vars, {metadata['edges']} edges")
    print(f"  {description}")

# Dataset 1: Empty graph (no edges)
print("\n=== Generating Synthetic Datasets ===\n")

structure = {i: [] for i in range(10)}
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("empty_graph", structure, data,
             "No edges - all variables independent")

# Dataset 2: Linear chain
structure = {i: [i-1] if i > 0 else [] for i in range(10)}
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("linear_chain", structure, data,
             "Chain: 0→1→2→3→4→5→6→7→8→9")

# Dataset 3: Star (one root to many children)
structure = {0: [], **{i: [0] for i in range(1, 10)}}
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("star", structure, data,
             "Star: 0 → {1,2,3,4,5,6,7,8,9}")

# Dataset 4: Collider (V-structure)
structure = {
    0: [], 1: [], 2: [0, 1],
    3: [], 4: [3], 5: [3],
    6: [], 7: [6], 8: [6], 9: []
}
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("colliders", structure, data,
             "Multiple V-structures: 0→2←1, 3→4, 3→5, 6→7, 6→8")

# Dataset 5: Fork (common cause)
structure = {
    0: [],
    1: [0], 2: [0], 3: [0],
    4: [1, 2], 5: [2, 3],
    6: [4], 7: [4], 8: [5], 9: [5]
}
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("fork", structure, data,
             "Common cause with convergence")

# Dataset 6: Dense random graph
np.random.seed(123)
structure = {}
for i in range(10):
    # Each node can have 0-4 parents from earlier nodes
    possible_parents = list(range(i))
    n_parents = np.random.randint(0, min(5, len(possible_parents) + 1))
    structure[i] = sorted(np.random.choice(possible_parents, n_parents, replace=False).tolist()) if possible_parents else []
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("dense_random", structure, data,
             "Dense random DAG (high connectivity)")

# Dataset 7: Sparse random graph
np.random.seed(456)
structure = {}
for i in range(10):
    possible_parents = list(range(i))
    # Max 1-2 parents
    n_parents = np.random.randint(0, min(2, len(possible_parents) + 1))
    structure[i] = sorted(np.random.choice(possible_parents, n_parents, replace=False).tolist()) if possible_parents else []
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("sparse_random", structure, data,
             "Sparse random DAG (low connectivity)")

# Dataset 8: Hierarchical layers
structure = {
    0: [], 1: [], 2: [],  # Layer 1 (roots)
    3: [0, 1], 4: [1, 2],  # Layer 2
    5: [3, 4], 6: [3, 4],  # Layer 3
    7: [5], 8: [6], 9: [5, 6]  # Layer 4
}
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("hierarchical", structure, data,
             "4-layer hierarchy")

# Dataset 9: Diamond pattern
structure = {
    0: [],
    1: [0], 2: [0], 3: [0],
    4: [1, 2], 5: [2, 3],
    6: [4, 5],
    7: [6], 8: [6], 9: [6]
}
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("diamond", structure, data,
             "Diamond: convergent and divergent paths")

# Dataset 10: Mixed with constant column
structure = {
    0: [],  # Constant (will be set to constant value)
    1: [0],
    2: [1], 3: [1],
    4: [2, 3],
    5: [],  # Independent root
    6: [5],
    7: [4, 6],
    8: [7], 9: [7]
}
data = generate_linear_gaussian_data(structure, 2000)
# Make column 0 constant (edge case)
data[:, 0] = 100.0
save_dataset("mixed_with_constant", structure, data,
             "Mixed structure with constant column (edge case)")

print("\n=== Summary ===")
print("Generated 10 synthetic datasets in synthetic_datasets/")
print("Each dataset has:")
print("  - 2000 samples")
print("  - 10 variables")
print("  - Known true DAG structure (saved in metadata)")
print("  - Linear Gaussian relationships")
