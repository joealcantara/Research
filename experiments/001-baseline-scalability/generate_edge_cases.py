#!/usr/bin/env python3
"""
Generate edge case datasets to test robustness.

These datasets test boundary conditions and special cases that might
reveal implementation bugs.
"""

import numpy as np
import json
from pathlib import Path

np.random.seed(42)

def save_edge_case(name, data, description):
    """Save edge case dataset."""
    output_dir = Path("edge_cases")
    output_dir.mkdir(exist_ok=True)

    np.savetxt(output_dir / f"{name}.csv", data, delimiter=',', fmt='%.6f')

    metadata = {
        "name": name,
        "description": description,
        "n_samples": data.shape[0],
        "n_vars": data.shape[1],
        "edge_case": True
    }

    with open(output_dir / f"{name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {name}: {data.shape[0]} samples × {data.shape[1]} vars")
    print(f"  {description}")

print("\n=== Generating Edge Case Datasets ===\n")

# Edge Case 1: All constant columns
data = np.full((100, 5), 100.0)
save_edge_case("all_constant", data,
               "All columns have same value (tests constant handling)")

# Edge Case 2: One constant, rest varying
data = np.random.randn(100, 5)
data[:, 2] = 42.0  # Column 2 is constant
save_edge_case("one_constant", data,
               "One constant column among varying ones")

# Edge Case 3: Perfect correlation
data = np.random.randn(100, 1)
data = np.column_stack([data, data, data * 2, data * -1, data + 1])
save_edge_case("perfect_correlation", data,
               "Perfectly correlated columns (linear dependencies)")

# Edge Case 4: Very small variance
data = np.random.randn(100, 5) * 0.0001  # Tiny variance
data += 1000  # Large mean
save_edge_case("tiny_variance", data,
               "Very small variance, large mean")

# Edge Case 5: Very large variance
data = np.random.randn(100, 5) * 10000  # Huge variance
save_edge_case("large_variance", data,
               "Very large variance")

# Edge Case 6: Extreme outliers
data = np.random.randn(100, 5)
data[0, :] = 1e6  # Extreme outlier
data[1, :] = -1e6
save_edge_case("extreme_outliers", data,
               "Contains extreme outliers (±1e6)")

# Edge Case 7: Minimal samples
data = np.random.randn(10, 5)  # Only 10 samples for 5 variables
save_edge_case("minimal_samples", data,
               "Very few samples (n=10) relative to variables")

# Edge Case 8: Single sample (extreme case)
data = np.random.randn(1, 5)
save_edge_case("single_sample", data,
               "Only 1 sample (tests degenerate case)")

# Edge Case 9: Binary-like data (0s and 1s)
data = np.random.choice([0.0, 1.0], size=(100, 5))
save_edge_case("binary_like", data,
               "Only 0s and 1s (mimics categorical)")

# Edge Case 10: Mixed: constant, correlated, and independent
data = np.zeros((100, 5))
data[:, 0] = 100.0  # Constant
data[:, 1] = np.random.randn(100)  # Independent
data[:, 2] = data[:, 1]  # Perfect copy of column 1
data[:, 3] = data[:, 1] * 2 + 5  # Linear transform of column 1
data[:, 4] = np.random.randn(100)  # Another independent
save_edge_case("mixed_degeneracies", data,
               "Mix of constant, correlated, and independent")

print("\n=== Summary ===")
print("Generated 10 edge case datasets in edge_cases/")
print("These test:")
print("  - Constant columns")
print("  - Perfect correlations")
print("  - Extreme values")
print("  - Tiny/large variance")
print("  - Minimal samples")
print("  - Degenerate cases")
