"""
Create SECOM dataset subsets with 50, 75, and 100 variables.

Randomly selects features from the full 590-feature SECOM dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Paths
SECOM_DATA = Path.home() / "Documents/Research/datasets/secom/secom.data"
SECOM_LABELS = Path.home() / "Documents/Research/datasets/secom/secom_labels.data"
OUTPUT_DIR = Path.home() / "Documents/Research/experiments/001-baseline-scalability/subsets"

# Load full dataset
print("Loading SECOM dataset...")
data = pd.read_csv(SECOM_DATA, sep=" ", header=None)
labels = pd.read_csv(SECOM_LABELS, sep=" ", header=None, usecols=[0])

print(f"Full dataset: {data.shape[0]} rows, {data.shape[1]} columns")

# Handle missing values - impute with column medians
print("Handling missing values...")
missing_before = data.isna().sum().sum()
print(f"  Missing values before: {missing_before}")

# Impute missing values with column medians
data = data.fillna(data.median())

# Verify no missing values remain
missing_after = data.isna().sum().sum()
print(f"  Missing values after imputation: {missing_after}")
print(f"  All rows retained: {data.shape[0]}")

# Set random seed for reproducibility
np.random.seed(42)

# Create subsets
for n_features in [50, 75, 100, 590]:
    print(f"\nCreating {n_features}-variable subset...")

    if n_features == 590:
        # Use all columns for full dataset
        selected_cols = list(range(data.shape[1]))
        subset = data
    else:
        # Randomly select n_features columns
        selected_cols = np.random.choice(data.shape[1], size=n_features, replace=False)
        selected_cols = sorted(selected_cols)  # Keep them ordered for consistency
        subset = data.iloc[:, selected_cols]

    # Save subset data and labels
    subset_path = OUTPUT_DIR / f"secom_{n_features}.csv"
    labels_path = OUTPUT_DIR / f"secom_{n_features}_labels.csv"

    subset.to_csv(subset_path, index=False, header=False)
    labels.to_csv(labels_path, index=False, header=False)

    print(f"  Saved: {subset_path}")
    print(f"  Shape: {subset.shape}")
    print(f"  Selected columns: {selected_cols[:10]}... (first 10)")

print("\nDone! Created 3 subsets in", OUTPUT_DIR)
