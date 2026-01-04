#!/usr/bin/env python3
"""Create a small 10-variable test subset for rapid testing."""

import pandas as pd
import numpy as np

# Load the 50-variable subset (already small)
data = pd.read_csv('subsets/secom_50.csv', header=None)
labels = pd.read_csv('subsets/secom_50_labels.csv', header=None)

# Take first 10 columns
test_data = data.iloc[:, :10]

# Save test subset
test_data.to_csv('subsets/secom_10.csv', index=False, header=False)
labels.to_csv('subsets/secom_10_labels.csv', index=False, header=False)

print(f"Created test subset: {test_data.shape[0]} rows × {test_data.shape[1]} columns")
print(f"Saved to: subsets/secom_10.csv and subsets/secom_10_labels.csv")

# Show some stats
print(f"\nColumn statistics:")
for i in range(test_data.shape[1]):
    col = test_data.iloc[:, i]
    n_unique = col.nunique()
    has_nan = col.isna().any()
    print(f"  Column {i}: {n_unique} unique values, NaN: {has_nan}")
