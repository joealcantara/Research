#!/usr/bin/env python3
"""
Generate test data for 3-variable chain: X1 → X2 → X3

This creates a dataset with known causal structure for validation.
Goal: Verify exhaustive search finds correct posterior distribution.
"""

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Sample size
n = 100

# Generate X1 ~ Bernoulli(0.5)
X1 = np.random.binomial(1, 0.5, n)

# Generate X2 | X1 with strong dependence
# P(X2=1|X1=1) = 0.8, P(X2=1|X1=0) = 0.2
X2 = np.where(X1 == 1,
              np.random.binomial(1, 0.8, n),
              np.random.binomial(1, 0.2, n))

# Generate X3 | X2 with strong dependence
# P(X3=1|X2=1) = 0.8, P(X3=1|X2=0) = 0.2
X3 = np.where(X2 == 1,
              np.random.binomial(1, 0.8, n),
              np.random.binomial(1, 0.2, n))

# Create DataFrame
df = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3
})

# Save to CSV
output_path = '../data/chain_3var.csv'
df.to_csv(output_path, index=False)

print(f"Generated {n} samples with structure X1 → X2 → X3")
print(f"Saved to: {output_path}")
print(f"\nData statistics:")
print(df.describe())
print(f"\nCorrelations:")
print(df.corr())
