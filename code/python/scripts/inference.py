"""
Bayesian inference for causal structure learning.

Functions for estimating parameters, computing likelihoods, and scoring
DAG structures against data using Bayesian inference.
"""

import numpy as np
import pandas as pd
from itertools import product


def _get_variable_values(df, var):
    """Helper function to get possible values for a variable."""
    unique_vals = sorted(df[var].unique())
    return unique_vals


def _get_variable_type(df, var, max_categorical=10):
    """
    Determine if a variable is binary, categorical, or continuous.

    Args:
        df: DataFrame
        var: Variable name
        max_categorical: Maximum unique values to consider categorical (default: 10)

    Returns:
        'binary', 'categorical', or 'continuous'
    """
    unique_vals = _get_variable_values(df, var)

    # Binary: exactly 2 values that are 0 and 1
    if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
        return 'binary'

    # Categorical: discrete values, not too many unique values
    if len(unique_vals) <= max_categorical and df[var].dtype in ['int64', 'int32', 'object', 'category']:
        return 'categorical'

    # Continuous: everything else
    return 'continuous'


def estimate_parameters(df, structure):
    """
    Estimate conditional probability tables/distributions from data.

    Given a DAG structure and data, learn the conditional probabilities
    for each variable given its parents. Handles binary, categorical, and continuous variables.

    Args:
        df: DataFrame with variables (binary, categorical, or continuous)
        structure: Dict mapping variables to their parent lists
                  e.g., {'X1': [], 'X2': ['X1'], 'X3': ['X2']}

    Returns:
        Dict of parameters for each variable:
        For binary variables:
        {
            'X1': {(): 0.6},                    # P(X1=1) = 0.6
            'X2': {(0,): 0.3, (1,): 0.9},       # P(X2=1 | X1=0) = 0.3, etc.
        }
        For categorical variables (e.g., values 1,2,3):
        {
            'species': {(): {1: 0.33, 2: 0.33, 3: 0.34}},  # Marginal probabilities
            'species': {(0,): {1: 0.5, 2: 0.3, 3: 0.2}}    # Conditional probabilities
        }
        For continuous variables:
        {
            'X1': {(): {'type': 'gaussian', 'mean': 5.8, 'std': 0.8}},  # Marginal
            'X2': {(): {'type': 'linear', 'coeffs': [1.2, 0.5], 'intercept': 0.3, 'std': 0.5}}  # Linear regression
        }
    """
    params = {}

    for var, parents in structure.items():
        var_type = _get_variable_type(df, var)

        if len(parents) == 0:
            # No parents: marginal distribution
            if var_type == 'binary':
                # Binary: P(var=1)
                count_1 = (df[var] == 1).sum()
                total = len(df)
                params[var] = {(): (count_1 + 1) / (total + 2)}

            elif var_type == 'categorical':
                # Categorical: P(var=k) for each value k
                unique_vals = _get_variable_values(df, var)
                total = len(df)
                num_values = len(unique_vals)
                prob_dist = {}
                for val in unique_vals:
                    count = (df[var] == val).sum()
                    prob_dist[val] = (count + 1) / (total + num_values)
                params[var] = {(): prob_dist}

            else:  # continuous
                # Continuous: Gaussian with empirical mean and std
                params[var] = {(): {
                    'type': 'gaussian',
                    'mean': df[var].mean(),
                    'std': df[var].std() + 1e-6  # Add small constant to avoid zero
                }}

        else:
            # Has parents
            if var_type == 'continuous':
                # Continuous variable: Linear regression
                X = df[parents].values
                y = df[var].values

                # Simple linear regression: y = X @ coeffs + intercept
                # Add intercept column
                X_with_intercept = np.column_stack([np.ones(len(X)), X])

                # Solve: coeffs = (X^T X)^{-1} X^T y
                try:
                    coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                    intercept = coeffs[0]
                    slopes = coeffs[1:]

                    # Compute residuals
                    predictions = X_with_intercept @ coeffs
                    residuals = y - predictions
                    residual_std = np.std(residuals) + 1e-6

                    params[var] = {(): {
                        'type': 'linear',
                        'intercept': intercept,
                        'coeffs': slopes,
                        'parents': parents,
                        'std': residual_std
                    }}
                except np.linalg.LinAlgError:
                    # Fallback if regression fails
                    params[var] = {(): {
                        'type': 'gaussian',
                        'mean': df[var].mean(),
                        'std': df[var].std() + 1e-6
                    }}

            else:
                # Binary or categorical with parents: conditional probability tables
                unique_vals = _get_variable_values(df, var) if var_type == 'categorical' else None

                # Get parent values - discretize continuous parents to avoid explosion
                parent_values = []
                for p in parents:
                    p_type = _get_variable_type(df, p)
                    if p_type == 'continuous':
                        # Discretize continuous parent using median
                        median = df[p].median()
                        parent_values.append([0, 1])  # Below/above median
                    else:
                        parent_values.append(_get_variable_values(df, p))

                all_combos = list(product(*parent_values))

                params[var] = {}
                for combo in all_combos:
                    # Filter data where parents match this combo
                    mask = pd.Series([True] * len(df))
                    for i, parent in enumerate(parents):
                        p_type = _get_variable_type(df, parent)
                        if p_type == 'continuous':
                            # Discretize continuous: 0 = below median, 1 = above median
                            median = df[parent].median()
                            if combo[i] == 0:
                                mask = mask & (df[parent] <= median)
                            else:
                                mask = mask & (df[parent] > median)
                        else:
                            mask = mask & (df[parent] == combo[i])

                    subset = df[mask]

                    if var_type == 'binary':
                        # Binary: P(var=1 | parents)
                        count_1 = (subset[var] == 1).sum()
                        total = len(subset)
                        params[var][combo] = (count_1 + 1) / (total + 2)
                    else:  # categorical
                        # Categorical: P(var=k | parents) for each value k
                        total = len(subset)
                        num_values = len(unique_vals)
                        prob_dist = {}
                        for val in unique_vals:
                            count = (subset[var] == val).sum()
                            prob_dist[val] = (count + 1) / (total + num_values)
                        params[var][combo] = prob_dist

    return params


def compute_log_likelihood(df, structure, params):
    """
    Compute log probability of the data given the structure and parameters.

    Calculates: log P(data | structure, params)

    This is the sum over all rows of log P(row | structure, params), where
    each row's probability is the product of P(each variable | its parents).

    Handles binary, categorical, and continuous variables.

    Args:
        df: DataFrame with variables (binary, categorical, or continuous)
        structure: DAG structure dict
        params: Parameters from estimate_parameters()

    Returns:
        Log-likelihood (float)
    """
    log_lik = 0.0

    for idx, row in df.iterrows():
        for var, parents in structure.items():
            var_type = _get_variable_type(df, var)

            # Get parent values for this row
            if len(parents) == 0:
                parent_combo = ()
            else:
                # Discretize continuous parents if needed (for categorical/binary variables)
                if var_type in ['binary', 'categorical']:
                    parent_combo = []
                    for p in parents:
                        p_type = _get_variable_type(df, p)
                        if p_type == 'continuous':
                            # Discretize: 0 if below median, 1 if above
                            median = df[p].median()
                            parent_combo.append(0 if row[p] <= median else 1)
                        else:
                            parent_combo.append(row[p])
                    parent_combo = tuple(parent_combo)
                else:
                    parent_combo = tuple(row[p] for p in parents)

            # Get parameter for this variable
            param_value = params[var].get(parent_combo, params[var].get((), None))
            if param_value is None:
                # Fallback for continuous variables
                param_value = params[var][()]

            # Determine how to compute likelihood based on parameter type
            if isinstance(param_value, dict) and 'type' in param_value:
                # Continuous variable
                observed_val = row[var]

                if param_value['type'] == 'gaussian':
                    # Gaussian: log N(x | mean, std^2)
                    mean = param_value['mean']
                    std = param_value['std']
                    log_lik += -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((observed_val - mean) / std)**2

                elif param_value['type'] == 'linear':
                    # Linear regression: predict, then Gaussian around prediction
                    parent_vals = [row[p] for p in param_value['parents']]
                    prediction = param_value['intercept'] + np.dot(param_value['coeffs'], parent_vals)
                    std = param_value['std']
                    log_lik += -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((observed_val - prediction) / std)**2

            elif isinstance(param_value, dict):
                # Categorical: look up probability for observed value
                observed_val = row[var]
                prob = param_value[observed_val]
                log_lik += np.log(prob)

            else:
                # Binary: param_value is P(var=1)
                if row[var] == 1:
                    log_lik += np.log(param_value)
                else:
                    log_lik += np.log(1 - param_value)

    return log_lik


def compute_posteriors(df, structures, lambda_penalty=2.0, store_params=False):
    """
    Compute posterior probability for each structure using Bayesian inference.

    For each structure:
    1. Estimate parameters (conditional probabilities)
    2. Compute log-likelihood (how well it explains the data)
    3. Apply complexity penalty (penalize more edges)
    4. Compute posterior = likelihood × prior

    Args:
        df: DataFrame with binary variables
        structures: List of DAG structures or dict of {name: structure}
        lambda_penalty: Cost per edge in log-likelihood units (default: 2.0)
        store_params: If True, store estimated parameters in results (needed for queries)

    Returns:
        Dict mapping structure names/indices to results:
        {
            'structure_name': {
                'log_likelihood': float,
                'edges': int,
                'log_prior': float,
                'log_posterior': float,
                'posterior': float,  # Normalized probability
                'params': dict  # Only if store_params=True
            }
        }
    """
    # Handle both list and dict inputs
    if isinstance(structures, list):
        structures_dict = {i: s for i, s in enumerate(structures)}
    else:
        structures_dict = structures

    results = {}

    for name, structure in structures_dict.items():
        params = estimate_parameters(df, structure)
        log_lik = compute_log_likelihood(df, structure, params)

        # Count edges for complexity penalty
        edges = sum(len(parents) for parents in structure.values())

        # Log posterior (unnormalized)
        log_prior = -lambda_penalty * edges
        log_posterior = log_lik + log_prior

        results[name] = {
            'log_likelihood': log_lik,
            'edges': edges,
            'log_prior': log_prior,
            'log_posterior': log_posterior
        }

        # Store parameters if requested (needed for probabilistic queries)
        if store_params:
            results[name]['params'] = params

    # Normalize to get probabilities
    log_posteriors = np.array([r['log_posterior'] for r in results.values()])

    # Subtract max for numerical stability
    log_posteriors_shifted = log_posteriors - log_posteriors.max()
    posteriors = np.exp(log_posteriors_shifted)
    posteriors = posteriors / posteriors.sum()

    # Add normalized posteriors to results
    for i, name in enumerate(results.keys()):
        results[name]['posterior'] = posteriors[i]

    return results


# Probabilistic Query Functions

def compute_joint_prob(assignment, structure, params):
    """
    Compute joint probability P(X1=x1, X2=x2, ...) under a structure.

    Joint probability = product of P(each variable | its parents)

    Args:
        assignment: Dict mapping variables to values, e.g., {'X1': 1, 'X2': 0, 'X3': 1}
        structure: DAG structure dict
        params: Parameters from estimate_parameters()

    Returns:
        Joint probability (float)
    """
    prob = 1.0

    for var, parents in structure.items():
        if len(parents) == 0:
            parent_combo = ()
        else:
            parent_combo = tuple(assignment[p] for p in parents)

        p_1 = params[var][parent_combo]

        if assignment[var] == 1:
            prob *= p_1
        else:
            prob *= (1 - p_1)

    return prob


def query_single_structure(query_var, query_val, evidence, structure, params):
    """
    Compute P(query_var = query_val | evidence) under a single structure.

    Uses enumeration over hidden variables to compute the conditional probability.

    Args:
        query_var: Variable to query (e.g., 'X3')
        query_val: Value to query (0 or 1)
        evidence: Dict of observed variables, e.g., {'X1': 1}
        structure: DAG structure dict
        params: Parameters from estimate_parameters()

    Returns:
        Conditional probability P(query_var = query_val | evidence)
    """
    # Get all variables in the structure
    variables = list(structure.keys())

    # Find hidden variables (not query, not evidence)
    hidden = [v for v in variables if v != query_var and v not in evidence]

    # Enumerate all combinations of hidden variables
    if len(hidden) == 0:
        hidden_combos = [{}]
    else:
        hidden_combos = [dict(zip(hidden, vals)) for vals in product([0, 1], repeat=len(hidden))]

    prob_query_and_evidence = 0.0
    prob_evidence = 0.0

    for hidden_vals in hidden_combos:
        # Compute P(query, evidence, hidden) for both query values
        for qval in [0, 1]:
            full_assignment = {**evidence, **hidden_vals, query_var: qval}
            p = compute_joint_prob(full_assignment, structure, params)

            prob_evidence += p
            if qval == query_val:
                prob_query_and_evidence += p

    # P(query | evidence) = P(query, evidence) / P(evidence)
    if prob_evidence == 0:
        return 0.5  # Fallback if no data

    return prob_query_and_evidence / prob_evidence


def query_with_uncertainty(query_var, query_val, evidence, structures, posterior_results):
    """
    Answer a query by averaging across all structures weighted by posterior probability.

    This implements Bayesian model averaging: instead of using a single best structure,
    we average predictions across all structures, weighted by how likely each structure is.

    Args:
        query_var: Variable to query (e.g., 'X3')
        query_val: Value to query (0 or 1)
        evidence: Dict of observed variables, e.g., {'X1': 1}
        structures: Dict of {name: structure}
        posterior_results: Results from compute_posteriors() with store_params=True

    Returns:
        Tuple of (weighted_answer, breakdown)
        - weighted_answer: Final probability averaged across all structures
        - breakdown: Dict with per-structure details
    """
    breakdown = {}
    weighted_answer = 0.0

    for name, structure in structures.items():
        params = posterior_results[name]['params']
        posterior = posterior_results[name]['posterior']

        # Compute answer under this structure
        answer = query_single_structure(query_var, query_val, evidence, structure, params)

        breakdown[name] = {
            'answer': answer,
            'posterior': posterior,
            'contribution': answer * posterior
        }

        weighted_answer += answer * posterior

    return weighted_answer, breakdown
