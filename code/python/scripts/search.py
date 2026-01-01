"""
Stochastic search algorithms for causal structure learning.

For problems where enumerating all DAG structures is intractable (n > 5 variables),
these algorithms provide efficient search strategies to find high-scoring structures.
"""

import numpy as np
import pandas as pd
from itertools import product


def sample_theories(scored_theories, k, temperature=1.0):
    """
    Sample k theories proportionally to their scores.

    Uses softmax sampling to favor high-scoring theories while maintaining diversity.

    Args:
        scored_theories: List of (theory, score) tuples
        k: Number of theories to sample
        temperature: Controls exploration (higher = more uniform sampling)

    Returns:
        List of sampled (theory, score) tuples
    """
    if len(scored_theories) <= k:
        return scored_theories

    theories, scores = zip(*scored_theories)
    scores = np.array(scores)

    # Convert log scores to probabilities using softmax
    scores_shifted = scores - scores.max()
    probs = np.exp(scores_shifted / temperature)
    probs = probs / probs.sum()

    # Sample without replacement
    indices = np.random.choice(
        len(theories),
        size=min(k, len(theories)),
        replace=False,
        p=probs
    )

    return [scored_theories[i] for i in indices]


def score_theory(df, parent_vars, target, lambda_penalty=2.0):
    """
    Score a theory: parent_vars → target.

    Computes log posterior = log likelihood + complexity penalty.

    Args:
        df: DataFrame with binary variables
        parent_vars: List of parent variable names (predictors)
        target: Target variable name
        lambda_penalty: Complexity penalty per parent (default: 2.0)

    Returns:
        Log posterior score (float)
    """
    if len(parent_vars) == 0:
        # No parents: just marginal probability
        p_1 = (df[target].sum() + 1) / (len(df) + 2)
        log_lik = (df[target] * np.log(p_1) + (1 - df[target]) * np.log(1 - p_1)).sum()
    else:
        log_lik = 0.0
        all_combos = list(product([0, 1], repeat=len(parent_vars)))

        for combo in all_combos:
            # Filter rows matching this parent combination
            mask = pd.Series([True] * len(df))
            for i, var in enumerate(parent_vars):
                mask = mask & (df[var] == combo[i])

            subset = df[mask]
            if len(subset) == 0:
                continue

            # P(target=1 | this parent combo)
            p_1 = (subset[target].sum() + 1) / (len(subset) + 2)

            # Add log likelihood for these rows
            log_lik += (subset[target] * np.log(p_1) + (1 - subset[target]) * np.log(1 - p_1)).sum()

    # Apply complexity penalty
    log_prior = -lambda_penalty * len(parent_vars)

    return log_lik + log_prior


def stochastic_beam_search(df, target, k=10, max_rounds=3, temperature=1.0, lambda_penalty=2.0, verbose=True):
    """
    Find theories that explain target using stochastic beam search.

    Instead of enumerating all possible parent sets, this algorithm:
    1. Starts with single-variable theories
    2. Scores each theory
    3. Samples top k theories (beam width)
    4. Expands by adding more parents
    5. Repeats until convergence or max_rounds

    This is much more efficient than enumeration for large variable sets:
    - n=3: 25 DAGs (enumeration fine)
    - n=5: 29,281 DAGs (enumeration slow)
    - n=6: 3.7M DAGs (beam search necessary)

    Args:
        df: DataFrame with binary variables
        target: Target variable to predict
        k: Beam width (number of theories to keep each round)
        max_rounds: Maximum search rounds (default: 3)
        temperature: Sampling temperature (default: 1.0)
        lambda_penalty: Complexity penalty (default: 2.0)
        verbose: Print progress (default: True)

    Returns:
        List of (theory, score) tuples sorted by score (best first)
    """
    predictors = [c for c in df.columns if c != target]

    # Round 0: Initialize with single-variable theories
    theories = [[v] for v in predictors]
    theories.append([])  # Also consider: nothing predicts target

    scored = [(t, score_theory(df, t, target, lambda_penalty)) for t in theories]
    current = sample_theories(scored, k, temperature)

    best_score = max(s for _, s in current)

    # Iterative expansion
    for round_num in range(max_rounds):
        # Expand each survivor by adding parents
        candidates = []

        for theory, old_score in current:
            # Keep the original theory
            candidates.append((theory, old_score))

            # Try adding each predictor not already in the theory
            for v in predictors:
                if v not in theory:
                    new_theory = theory + [v]
                    new_score = score_theory(df, new_theory, target, lambda_penalty)
                    candidates.append((new_theory, new_score))

        # Remove duplicate theories
        seen = set()
        unique_candidates = []
        for theory, score in candidates:
            key = tuple(sorted(theory))
            if key not in seen:
                seen.add(key)
                unique_candidates.append((theory, score))

        # Sample for next round
        current = sample_theories(unique_candidates, k, temperature)

        new_best = max(s for _, s in current)

        if verbose:
            print(f"Round {round_num + 1}: best score = {new_best:.2f} ({len(unique_candidates)} candidates)")

        # Early stopping if no improvement
        if new_best <= best_score + 0.1:
            if verbose:
                print(f"Converged (no improvement > 0.1)")
            break

        best_score = new_best

    # Return sorted by score (best first)
    current.sort(key=lambda x: -x[1])
    return current
