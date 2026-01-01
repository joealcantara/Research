"""
Data generation for Bayesian causal discovery experiments.

Generates synthetic data from known causal structures (chain, fork, collider)
for testing and validating causal inference algorithms.
"""

import numpy as np
import pandas as pd


def generate_chain_data(n_samples, seed=42):
    """
    Generate data from a Chain structure: X1 → X2 → X3

    Parameters:
    - P(X1 = 1) = 0.5
    - P(X2 = 1 | X1 = 1) = 0.9
    - P(X2 = 1 | X1 = 0) = 0.2
    - P(X3 = 1 | X2 = 1) = 0.8
    - P(X3 = 1 | X2 = 0) = 0.3
    """
    np.random.seed(seed)

    data = []

    for i in range(n_samples):
        # Step 1: Sample X1
        x1 = 1 if np.random.random() < 0.5 else 0

        # Step 2: Sample X2 given X1
        if x1 == 1:
            x2 = 1 if np.random.random() < 0.9 else 0
        else:
            x2 = 1 if np.random.random() < 0.2 else 0

        # Step 3: Sample X3 given X2
        if x2 == 1:
            x3 = 1 if np.random.random() < 0.8 else 0
        else:
            x3 = 1 if np.random.random() < 0.3 else 0

        data.append([x1, x2, x3])

    # Return as dataframe with anonymous column names
    return pd.DataFrame(data, columns=['X1', 'X2', 'X3'])


def generate_fork_data(n_samples, seed=42):
    """
    Fork structure: X2 ← X1 → X3

    X1 is a common cause of both X2 and X3
    X2 and X3 are correlated, but only because they share a cause
    """
    np.random.seed(seed)

    data = []
    for i in range(n_samples):
        # X1 is a root node
        x1 = 1 if np.random.random() < 0.5 else 0

        # X2 depends on X1
        if x1 == 1:
            x2 = 1 if np.random.random() < 0.9 else 0
        else:
            x2 = 1 if np.random.random() < 0.2 else 0

        # X3 depends on X1 (not X2)
        if x1 == 1:
            x3 = 1 if np.random.random() < 0.8 else 0
        else:
            x3 = 1 if np.random.random() < 0.3 else 0

        data.append([x1, x2, x3])

    return pd.DataFrame(data, columns=['X1', 'X2', 'X3'])


def generate_collider_data(n_samples, seed=42):
    """
    Collider structure: X1 → X2 ← X3

    X1 and X3 are independent causes of X2
    X1 and X3 are uncorrelated... unless you condition on X2
    """
    np.random.seed(seed)

    data = []
    for i in range(n_samples):
        # X1 is a root node
        x1 = 1 if np.random.random() < 0.5 else 0

        # X3 is also a root node (independent of X1)
        x3 = 1 if np.random.random() < 0.5 else 0

        # X2 depends on both X1 and X3
        if x1 == 1 and x3 == 1:
            x2 = 1 if np.random.random() < 0.95 else 0
        elif x1 == 1 and x3 == 0:
            x2 = 1 if np.random.random() < 0.7 else 0
        elif x1 == 0 and x3 == 1:
            x2 = 1 if np.random.random() < 0.7 else 0
        else:  # x1 == 0 and x3 == 0
            x2 = 1 if np.random.random() < 0.1 else 0

        data.append([x1, x2, x3])

    return pd.DataFrame(data, columns=['X1', 'X2', 'X3'])
