"""
Tests for data generation and DAG enumeration functions.

Run with: pytest tests.py
Or simply: python tests.py
"""

import numpy as np
import pandas as pd
from scripts.generate_data import generate_chain_data, generate_fork_data, generate_collider_data
from scripts.dag_utils import generate_all_dags, count_edges, is_dag
from scripts.inference import (
    estimate_parameters, compute_log_likelihood, compute_posteriors,
    compute_joint_prob, query_single_structure, query_with_uncertainty
)
from scripts.search import sample_theories, score_theory, stochastic_beam_search


def test_chain_data_shape():
    """Test that chain data has correct shape."""
    df = generate_chain_data(100, seed=42)
    assert df.shape == (100, 3), f"Expected (100, 3), got {df.shape}"
    print("✓ Chain data shape test passed")


def test_chain_data_columns():
    """Test that chain data has correct column names."""
    df = generate_chain_data(100, seed=42)
    assert list(df.columns) == ['X1', 'X2', 'X3'], f"Expected ['X1', 'X2', 'X3'], got {list(df.columns)}"
    print("✓ Chain data columns test passed")


def test_chain_data_binary():
    """Test that chain data contains only binary values."""
    df = generate_chain_data(100, seed=42)
    for col in df.columns:
        unique_vals = set(df[col].unique())
        assert unique_vals.issubset({0, 1}), f"Column {col} has non-binary values: {unique_vals}"
    print("✓ Chain data binary values test passed")


def test_chain_data_deterministic():
    """Test that same seed produces same data."""
    df1 = generate_chain_data(100, seed=42)
    df2 = generate_chain_data(100, seed=42)
    assert df1.equals(df2), "Same seed should produce identical data"
    print("✓ Chain data deterministic test passed")


def test_fork_data_shape():
    """Test that fork data has correct shape."""
    df = generate_fork_data(100, seed=42)
    assert df.shape == (100, 3), f"Expected (100, 3), got {df.shape}"
    print("✓ Fork data shape test passed")


def test_fork_data_columns():
    """Test that fork data has correct column names."""
    df = generate_fork_data(100, seed=42)
    assert list(df.columns) == ['X1', 'X2', 'X3'], f"Expected ['X1', 'X2', 'X3'], got {list(df.columns)}"
    print("✓ Fork data columns test passed")


def test_fork_data_binary():
    """Test that fork data contains only binary values."""
    df = generate_fork_data(100, seed=42)
    for col in df.columns:
        unique_vals = set(df[col].unique())
        assert unique_vals.issubset({0, 1}), f"Column {col} has non-binary values: {unique_vals}"
    print("✓ Fork data binary values test passed")


def test_fork_data_deterministic():
    """Test that same seed produces same data."""
    df1 = generate_fork_data(100, seed=42)
    df2 = generate_fork_data(100, seed=42)
    assert df1.equals(df2), "Same seed should produce identical data"
    print("✓ Fork data deterministic test passed")


def test_collider_data_shape():
    """Test that collider data has correct shape."""
    df = generate_collider_data(100, seed=42)
    assert df.shape == (100, 3), f"Expected (100, 3), got {df.shape}"
    print("✓ Collider data shape test passed")


def test_collider_data_columns():
    """Test that collider data has correct column names."""
    df = generate_collider_data(100, seed=42)
    assert list(df.columns) == ['X1', 'X2', 'X3'], f"Expected ['X1', 'X2', 'X3'], got {list(df.columns)}"
    print("✓ Collider data columns test passed")


def test_collider_data_binary():
    """Test that collider data contains only binary values."""
    df = generate_collider_data(100, seed=42)
    for col in df.columns:
        unique_vals = set(df[col].unique())
        assert unique_vals.issubset({0, 1}), f"Column {col} has non-binary values: {unique_vals}"
    print("✓ Collider data binary values test passed")


def test_collider_data_deterministic():
    """Test that same seed produces same data."""
    df1 = generate_collider_data(100, seed=42)
    df2 = generate_collider_data(100, seed=42)
    assert df1.equals(df2), "Same seed should produce identical data"
    print("✓ Collider data deterministic test passed")


def test_chain_structure():
    """Test that chain data exhibits expected statistical structure."""
    # Generate large sample to test statistical properties
    df = generate_chain_data(1000, seed=42)

    # In a chain X1 → X2 → X3:
    # - X1 should be roughly 50% ones (marginal)
    # - X2 should be correlated with X1
    # - X3 should be correlated with X2

    p_x1 = df['X1'].mean()
    assert 0.4 < p_x1 < 0.6, f"P(X1=1) should be ~0.5, got {p_x1}"

    # Check X1-X2 correlation exists
    corr_12 = df['X1'].corr(df['X2'])
    assert corr_12 > 0.3, f"X1 and X2 should be correlated in chain, got {corr_12}"

    # Check X2-X3 correlation exists
    corr_23 = df['X2'].corr(df['X3'])
    assert corr_23 > 0.3, f"X2 and X3 should be correlated in chain, got {corr_23}"

    print("✓ Chain structure test passed")


def test_collider_independence():
    """Test that collider data shows X1 and X3 independence."""
    # In collider X1 → X2 ← X3:
    # - X1 and X3 should be independent (uncorrelated)

    df = generate_collider_data(1000, seed=42)

    corr_13 = df['X1'].corr(df['X3'])
    assert abs(corr_13) < 0.15, f"X1 and X3 should be independent in collider, got correlation {corr_13}"

    print("✓ Collider independence test passed")


# DAG Generation Tests

def test_generate_all_dags_count():
    """Test that we generate exactly 25 DAGs for 3 variables."""
    variables = ['X1', 'X2', 'X3']
    dags = generate_all_dags(variables)
    assert len(dags) == 25, f"Expected 25 DAGs for 3 variables, got {len(dags)}"
    print("✓ DAG count test passed (25 DAGs for 3 variables)")


def test_generate_all_dags_structure():
    """Test that each DAG has correct structure format."""
    variables = ['X1', 'X2', 'X3']
    dags = generate_all_dags(variables)

    for dag in dags:
        # Each DAG should be a dict
        assert isinstance(dag, dict), "DAG should be a dictionary"

        # Should have all variables as keys
        assert set(dag.keys()) == set(variables), f"DAG should have keys {variables}"

        # Each value should be a list
        for var, parents in dag.items():
            assert isinstance(parents, list), f"Parents of {var} should be a list"

            # Parents should be valid variable names
            for parent in parents:
                assert parent in variables, f"Parent {parent} should be in {variables}"

    print("✓ DAG structure format test passed")


def test_generate_all_dags_acyclic():
    """Test that all generated DAGs are acyclic."""
    variables = ['X1', 'X2', 'X3']
    dags = generate_all_dags(variables)

    for dag in dags:
        # Convert to edge list
        edges = []
        for i, var in enumerate(variables):
            parents = dag[var]
            for parent in parents:
                j = variables.index(parent)
                edges.append((j, i))

        # Check acyclic
        assert is_dag(edges, len(variables)), f"DAG {dag} should be acyclic"

    print("✓ All generated DAGs are acyclic test passed")


def test_count_edges():
    """Test edge counting for various structures."""
    # Independent structure (no edges)
    independent = {'X1': [], 'X2': [], 'X3': []}
    assert count_edges(independent) == 0, "Independent structure should have 0 edges"

    # Chain (2 edges)
    chain = {'X1': [], 'X2': ['X1'], 'X3': ['X2']}
    assert count_edges(chain) == 2, "Chain should have 2 edges"

    # Fork (2 edges)
    fork = {'X1': [], 'X2': ['X1'], 'X3': ['X1']}
    assert count_edges(fork) == 2, "Fork should have 2 edges"

    # Collider (2 edges)
    collider = {'X1': [], 'X2': ['X1', 'X3'], 'X3': []}
    assert count_edges(collider) == 2, "Collider should have 2 edges"

    # Full (3 edges)
    full = {'X1': [], 'X2': ['X1'], 'X3': ['X1', 'X2']}
    assert count_edges(full) == 3, "Full structure should have 3 edges"

    print("✓ Edge counting test passed")


def test_is_dag_valid():
    """Test that is_dag correctly identifies valid DAGs."""
    # Empty graph (no edges) - valid DAG
    assert is_dag([], 3), "Empty graph should be a valid DAG"

    # Single edge - valid DAG
    assert is_dag([(0, 1)], 3), "Single edge should be a valid DAG"

    # Chain: 0 → 1 → 2 - valid DAG
    assert is_dag([(0, 1), (1, 2)], 3), "Chain should be a valid DAG"

    # Fork: 1 → 0, 1 → 2 - valid DAG
    assert is_dag([(1, 0), (1, 2)], 3), "Fork should be a valid DAG"

    print("✓ is_dag valid structures test passed")


def test_is_dag_cycle():
    """Test that is_dag correctly detects cycles."""
    # Simple cycle: 0 → 1 → 0
    assert not is_dag([(0, 1), (1, 0)], 2), "Simple cycle should be detected"

    # Three-node cycle: 0 → 1 → 2 → 0
    assert not is_dag([(0, 1), (1, 2), (2, 0)], 3), "Three-node cycle should be detected"

    # Self-loop: 0 → 0
    assert not is_dag([(0, 0)], 1), "Self-loop should be detected"

    print("✓ is_dag cycle detection test passed")


def test_known_structures_present():
    """Test that known structures (chain, fork, collider) are in the generated set."""
    variables = ['X1', 'X2', 'X3']
    dags = generate_all_dags(variables)

    # Define known structures
    chain = {'X1': [], 'X2': ['X1'], 'X3': ['X2']}
    fork = {'X1': [], 'X2': ['X1'], 'X3': ['X1']}
    collider = {'X1': [], 'X2': ['X1', 'X3'], 'X3': []}
    independent = {'X1': [], 'X2': [], 'X3': []}

    # Check each is present
    assert chain in dags, "Chain structure should be in generated DAGs"
    assert fork in dags, "Fork structure should be in generated DAGs"
    assert collider in dags, "Collider structure should be in generated DAGs"
    assert independent in dags, "Independent structure should be in generated DAGs"

    print("✓ Known structures present test passed")


# Bayesian Inference Tests

def test_estimate_parameters_marginal():
    """Test parameter estimation for variables with no parents."""
    # Create simple data
    df = pd.DataFrame({
        'X1': [1, 1, 0, 1, 0],  # 3 ones out of 5 = 60%
        'X2': [1, 0, 1, 1, 1],
        'X3': [0, 0, 0, 1, 0]
    })

    structure = {'X1': [], 'X2': [], 'X3': []}
    params = estimate_parameters(df, structure)

    # With smoothing: (3+1)/(5+2) = 4/7 ≈ 0.571
    p_x1 = params['X1'][()]
    assert 0.5 < p_x1 < 0.65, f"P(X1=1) should be ~0.57 with smoothing, got {p_x1}"

    # X2: 4 ones, (4+1)/(5+2) = 5/7 ≈ 0.714
    p_x2 = params['X2'][()]
    assert 0.65 < p_x2 < 0.8, f"P(X2=1) should be ~0.71 with smoothing, got {p_x2}"

    print("✓ Parameter estimation (marginal) test passed")


def test_estimate_parameters_conditional():
    """Test parameter estimation for variables with parents."""
    # Create data where X2 depends on X1
    df = pd.DataFrame({
        'X1': [0, 0, 1, 1, 1, 1],
        'X2': [0, 0, 1, 1, 1, 0]  # When X1=1: mostly 1, When X1=0: mostly 0
    })

    structure = {'X1': [], 'X2': ['X1']}
    params = estimate_parameters(df, structure)

    # P(X2=1 | X1=0): 0 ones out of 2, with smoothing (0+1)/(2+2) = 1/4 = 0.25
    p_x2_given_x1_0 = params['X2'][(0,)]
    assert p_x2_given_x1_0 == 0.25, f"P(X2=1|X1=0) should be 0.25, got {p_x2_given_x1_0}"

    # P(X2=1 | X1=1): 3 ones out of 4, with smoothing (3+1)/(4+2) = 4/6 ≈ 0.667
    p_x2_given_x1_1 = params['X2'][(1,)]
    assert abs(p_x2_given_x1_1 - 2/3) < 0.01, f"P(X2=1|X1=1) should be ~0.667, got {p_x2_given_x1_1}"

    print("✓ Parameter estimation (conditional) test passed")


def test_compute_log_likelihood():
    """Test log-likelihood computation."""
    # Simple deterministic data
    df = pd.DataFrame({
        'X1': [1, 1, 0, 0],
        'X2': [1, 1, 0, 0]  # X2 = X1 (perfect correlation)
    })

    # Structure where X2 depends on X1 should have higher likelihood
    # than independent structure
    chain = {'X1': [], 'X2': ['X1']}
    params_chain = estimate_parameters(df, chain)
    ll_chain = compute_log_likelihood(df, chain, params_chain)

    independent = {'X1': [], 'X2': []}
    params_indep = estimate_parameters(df, independent)
    ll_indep = compute_log_likelihood(df, independent, params_indep)

    # Chain should have higher likelihood (less negative)
    assert ll_chain > ll_indep, f"Chain likelihood ({ll_chain}) should be > independent ({ll_indep})"

    print("✓ Log-likelihood computation test passed")


def test_compute_posteriors_format():
    """Test that compute_posteriors returns correct format."""
    df = generate_chain_data(100, seed=42)

    structures = {
        'chain': {'X1': [], 'X2': ['X1'], 'X3': ['X2']},
        'fork': {'X1': [], 'X2': ['X1'], 'X3': ['X1']},
        'independent': {'X1': [], 'X2': [], 'X3': []}
    }

    results = compute_posteriors(df, structures, lambda_penalty=2.0)

    # Check all structures are in results
    assert set(results.keys()) == set(structures.keys()), "Results should have all structure names"

    # Check each result has required fields
    for name, result in results.items():
        assert 'log_likelihood' in result, f"{name} should have log_likelihood"
        assert 'edges' in result, f"{name} should have edges"
        assert 'log_prior' in result, f"{name} should have log_prior"
        assert 'log_posterior' in result, f"{name} should have log_posterior"
        assert 'posterior' in result, f"{name} should have posterior"

    # Check posteriors sum to 1
    total_posterior = sum(r['posterior'] for r in results.values())
    assert abs(total_posterior - 1.0) < 1e-6, f"Posteriors should sum to 1, got {total_posterior}"

    print("✓ Posterior computation format test passed")


def test_compute_posteriors_true_structure_wins():
    """Test that true structure gets highest posterior on chain data."""
    # Generate data from chain structure
    df = generate_chain_data(200, seed=42)

    structures = {
        'chain': {'X1': [], 'X2': ['X1'], 'X3': ['X2']},
        'fork': {'X1': [], 'X2': ['X1'], 'X3': ['X1']},
        'collider': {'X1': [], 'X2': ['X1', 'X3'], 'X3': []},
        'independent': {'X1': [], 'X2': [], 'X3': []}
    }

    results = compute_posteriors(df, structures, lambda_penalty=2.0)

    # Chain should have highest posterior
    posteriors = {name: r['posterior'] for name, r in results.items()}
    best_structure = max(posteriors, key=posteriors.get)

    assert best_structure == 'chain', f"Chain should have highest posterior, but {best_structure} won with {posteriors}"

    # Chain posterior should be substantial (> 50%)
    assert posteriors['chain'] > 0.5, f"Chain posterior should be > 0.5, got {posteriors['chain']}"

    print("✓ True structure wins test passed")


def test_compute_posteriors_penalty():
    """Test that complexity penalty affects results."""
    df = generate_chain_data(100, seed=42)

    chain = {'X1': [], 'X2': ['X1'], 'X3': ['X2']}
    full = {'X1': [], 'X2': ['X1'], 'X3': ['X1', 'X2']}

    structures = {'chain': chain, 'full': full}

    # With low penalty, full might compete with chain
    results_low = compute_posteriors(df, structures, lambda_penalty=0.5)

    # With high penalty, chain should dominate (fewer edges)
    results_high = compute_posteriors(df, structures, lambda_penalty=5.0)

    # Chain's advantage should increase with higher penalty
    chain_advantage_low = results_low['chain']['posterior'] - results_low['full']['posterior']
    chain_advantage_high = results_high['chain']['posterior'] - results_high['full']['posterior']

    assert chain_advantage_high > chain_advantage_low, \
        f"Higher penalty should favor simpler structure more. Low: {chain_advantage_low}, High: {chain_advantage_high}"

    print("✓ Complexity penalty test passed")


# Probabilistic Query Tests

def test_compute_joint_prob():
    """Test joint probability computation."""
    # Simple structure: X1 → X2 (chain)
    structure = {'X1': [], 'X2': ['X1']}

    # Simple parameters
    params = {
        'X1': {(): 0.5},  # P(X1=1) = 0.5
        'X2': {(0,): 0.2, (1,): 0.8}  # P(X2=1|X1=0) = 0.2, P(X2=1|X1=1) = 0.8
    }

    # Test P(X1=1, X2=1) = P(X1=1) * P(X2=1|X1=1) = 0.5 * 0.8 = 0.4
    assignment = {'X1': 1, 'X2': 1}
    prob = compute_joint_prob(assignment, structure, params)
    assert abs(prob - 0.4) < 1e-6, f"P(X1=1, X2=1) should be 0.4, got {prob}"

    # Test P(X1=0, X2=1) = P(X1=0) * P(X2=1|X1=0) = 0.5 * 0.2 = 0.1
    assignment = {'X1': 0, 'X2': 1}
    prob = compute_joint_prob(assignment, structure, params)
    assert abs(prob - 0.1) < 1e-6, f"P(X1=0, X2=1) should be 0.1, got {prob}"

    # Test P(X1=1, X2=0) = P(X1=1) * P(X2=0|X1=1) = 0.5 * 0.2 = 0.1
    assignment = {'X1': 1, 'X2': 0}
    prob = compute_joint_prob(assignment, structure, params)
    assert abs(prob - 0.1) < 1e-6, f"P(X1=1, X2=0) should be 0.1, got {prob}"

    print("✓ Joint probability computation test passed")


def test_query_single_structure_simple():
    """Test querying a single structure with simple probabilities."""
    # Structure: X1 → X2
    structure = {'X1': [], 'X2': ['X1']}

    params = {
        'X1': {(): 0.5},
        'X2': {(0,): 0.2, (1,): 0.8}
    }

    # Query: P(X2=1 | X1=1)
    # This should be 0.8 directly from the CPT
    prob = query_single_structure('X2', 1, {'X1': 1}, structure, params)
    assert abs(prob - 0.8) < 1e-6, f"P(X2=1|X1=1) should be 0.8, got {prob}"

    # Query: P(X2=1 | X1=0)
    # This should be 0.2 directly from the CPT
    prob = query_single_structure('X2', 1, {'X1': 0}, structure, params)
    assert abs(prob - 0.2) < 1e-6, f"P(X2=1|X1=0) should be 0.2, got {prob}"

    print("✓ Single structure query (simple) test passed")


def test_query_single_structure_marginalization():
    """Test querying with marginalization over hidden variables."""
    # Structure: X1 → X2 → X3 (chain)
    structure = {'X1': [], 'X2': ['X1'], 'X3': ['X2']}

    params = {
        'X1': {(): 0.5},
        'X2': {(0,): 0.2, (1,): 0.8},
        'X3': {(0,): 0.3, (1,): 0.7}
    }

    # Query: P(X3=1 | X1=1)
    # Need to marginalize over X2:
    # P(X3=1|X1=1) = P(X3=1|X2=0)*P(X2=0|X1=1) + P(X3=1|X2=1)*P(X2=1|X1=1)
    #              = 0.3 * 0.2 + 0.7 * 0.8
    #              = 0.06 + 0.56 = 0.62
    prob = query_single_structure('X3', 1, {'X1': 1}, structure, params)
    assert abs(prob - 0.62) < 1e-6, f"P(X3=1|X1=1) should be 0.62, got {prob}"

    print("✓ Single structure query (marginalization) test passed")


def test_query_with_uncertainty():
    """Test Bayesian model averaging across multiple structures."""
    # Generate chain data
    df = generate_chain_data(100, seed=42)

    # Two competing structures
    structures = {
        'chain': {'X1': [], 'X2': ['X1'], 'X3': ['X2']},
        'independent': {'X1': [], 'X2': [], 'X3': []}
    }

    # Compute posteriors with parameters stored
    results = compute_posteriors(df, structures, lambda_penalty=2.0, store_params=True)

    # Query: P(X3=1 | X1=1)
    weighted_answer, breakdown = query_with_uncertainty('X3', 1, {'X1': 1}, structures, results)

    # Checks:
    # 1. Weighted answer should be a valid probability
    assert 0 <= weighted_answer <= 1, f"Weighted answer should be in [0,1], got {weighted_answer}"

    # 2. Breakdown should have entries for both structures
    assert 'chain' in breakdown, "Breakdown should include 'chain'"
    assert 'independent' in breakdown, "Breakdown should include 'independent'"

    # 3. Each breakdown entry should have required fields
    for name in structures.keys():
        assert 'answer' in breakdown[name], f"{name} should have 'answer'"
        assert 'posterior' in breakdown[name], f"{name} should have 'posterior'"
        assert 'contribution' in breakdown[name], f"{name} should have 'contribution'"

    # 4. Weighted answer should equal sum of contributions
    total_contribution = sum(b['contribution'] for b in breakdown.values())
    assert abs(weighted_answer - total_contribution) < 1e-6, \
        f"Weighted answer ({weighted_answer}) should equal sum of contributions ({total_contribution})"

    print("✓ Query with uncertainty test passed")


def test_query_with_uncertainty_chain_data():
    """Test that queries on chain data give reasonable results."""
    # Generate chain data with more samples for stability
    df = generate_chain_data(200, seed=42)

    structures = {
        'chain': {'X1': [], 'X2': ['X1'], 'X3': ['X2']},
        'fork': {'X1': [], 'X2': ['X1'], 'X3': ['X1']},
        'independent': {'X1': [], 'X2': [], 'X3': []}
    }

    results = compute_posteriors(df, structures, lambda_penalty=2.0, store_params=True)

    # Query: P(X3=1 | X1=1)
    # In chain data, X1 influences X3 through X2, so this should be > 0.5
    prob_x3_given_x1_1, _ = query_with_uncertainty('X3', 1, {'X1': 1}, structures, results)

    # Query: P(X3=1 | X1=0)
    # Should be < P(X3=1 | X1=1) for chain data
    prob_x3_given_x1_0, _ = query_with_uncertainty('X3', 1, {'X1': 0}, structures, results)

    # P(X3=1|X1=1) should be greater than P(X3=1|X1=0) for chain data
    assert prob_x3_given_x1_1 > prob_x3_given_x1_0, \
        f"For chain data, P(X3=1|X1=1)={prob_x3_given_x1_1} should be > P(X3=1|X1=0)={prob_x3_given_x1_0}"

    print("✓ Query with uncertainty (chain data) test passed")


# Stochastic Search Tests

def test_score_theory_empty():
    """Test scoring a theory with no parents."""
    df = pd.DataFrame({
        'X1': [1, 1, 0, 1, 0],
        'X2': [1, 0, 1, 1, 1]
    })

    # Score theory: (nothing) → X2
    score = score_theory(df, [], 'X2', lambda_penalty=2.0)

    # Should just be marginal likelihood (no penalty for 0 parents)
    assert score < 0, "Score should be negative (log probability)"
    assert score > -100, "Score should be reasonable magnitude"

    print("✓ Score theory (empty parents) test passed")


def test_score_theory_single_parent():
    """Test scoring a theory with one parent."""
    df = generate_chain_data(100, seed=42)

    # Score theory: X1 → X2
    score_x1 = score_theory(df, ['X1'], 'X2', lambda_penalty=2.0)

    # Score theory: (nothing) → X2
    score_empty = score_theory(df, [], 'X2', lambda_penalty=2.0)

    # X1→X2 should score better than nothing (X1 causes X2 in chain data)
    # Even with penalty of -2.0 for one edge
    assert score_x1 > score_empty, f"X1→X2 ({score_x1}) should score better than empty ({score_empty})"

    print("✓ Score theory (single parent) test passed")


def test_sample_theories():
    """Test sampling theories proportionally to scores."""
    # Create scored theories with clear differences
    theories = [
        (['X1'], -10.0),  # Best
        (['X2'], -20.0),  # Medium
        (['X3'], -30.0),  # Worst
    ]

    # Sample 2 theories (should favor high scores)
    sampled = sample_theories(theories, k=2, temperature=1.0)

    # Should return 2 theories
    assert len(sampled) == 2, f"Should sample 2 theories, got {len(sampled)}"

    # Check format
    for theory, score in sampled:
        assert isinstance(theory, list), "Theory should be a list"
        assert isinstance(score, float), "Score should be float"

    print("✓ Sample theories test passed")


def test_stochastic_beam_search_chain():
    """Test beam search on chain data identifies X2→X3."""
    df = generate_chain_data(100, seed=42)

    # Search for theories explaining X3
    results = stochastic_beam_search(df, target='X3', k=5, max_rounds=3, verbose=False)

    # Should return theories
    assert len(results) > 0, "Should find at least one theory"

    # Best theory should involve X2 (direct cause in chain)
    best_theory, best_score = results[0]

    # X2 should be in the best theory (X2→X3 in chain structure)
    assert 'X2' in best_theory, f"Best theory {best_theory} should include X2 (direct cause)"

    print("✓ Beam search (chain data) test passed")


def test_stochastic_beam_search_fork():
    """Test beam search on fork data identifies X1→X3."""
    df = generate_fork_data(100, seed=42)

    # Search for theories explaining X3
    results = stochastic_beam_search(df, target='X3', k=5, max_rounds=3, verbose=False)

    # Best theory should involve X1 (direct cause in fork)
    best_theory, best_score = results[0]

    # X1 should be in the best theory (X1→X3 in fork structure)
    assert 'X1' in best_theory, f"Best theory {best_theory} should include X1 (direct cause)"

    print("✓ Beam search (fork data) test passed")


# Categorical variable tests (multi-valued)

def test_categorical_no_parents():
    """Test categorical variable with no parents."""
    df = pd.DataFrame({
        'species': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] * 10
    })

    structure = {'species': []}
    params = estimate_parameters(df, structure)

    # Should be probability distribution over values
    assert () in params['species']
    assert isinstance(params['species'][()], dict)
    assert 1 in params['species'][()]
    assert 2 in params['species'][()]
    assert 3 in params['species'][()]

    # Probabilities should sum to 1
    prob_sum = sum(params['species'][()].values())
    assert abs(prob_sum - 1.0) < 0.01

    print("✓ Categorical variable (no parents) test passed")


def test_categorical_with_categorical_parents():
    """Test categorical variable with categorical parents."""
    # Create data where color depends on size
    df = pd.DataFrame({
        'size': [1, 1, 2, 2, 3, 3] * 10,  # Small, Medium, Large
        'color': [1, 1, 2, 2, 3, 3] * 10  # Red, Green, Blue
    })

    structure = {'size': [], 'color': ['size']}
    params = estimate_parameters(df, structure)

    # Color should have CPT with one entry per size value
    assert len(params['color']) == 3  # 3 size values

    # Each entry should be a probability distribution
    for size_val in [1, 2, 3]:
        assert (size_val,) in params['color']
        assert isinstance(params['color'][(size_val,)], dict)
        prob_sum = sum(params['color'][(size_val,)].values())
        assert abs(prob_sum - 1.0) < 0.01

    print("✓ Categorical variable (with categorical parents) test passed")


def test_categorical_likelihood():
    """Test log-likelihood computation for categorical variables."""
    df = pd.DataFrame({
        'x': [1, 2, 3, 1, 2, 3, 1, 2, 3] * 10
    })

    structure = {'x': []}
    params = estimate_parameters(df, structure)
    log_lik = compute_log_likelihood(df, structure, params)

    # Log-likelihood should be finite and negative
    assert np.isfinite(log_lik)
    assert log_lik < 0

    print("✓ Categorical variable likelihood test passed")


def test_mixed_binary_categorical():
    """Test mixed structure with binary and categorical variables."""
    # Binary influences categorical
    df = pd.DataFrame({
        'binary': [0, 0, 1, 1] * 25,
        'categorical': [1, 1, 2, 3] * 25  # Different distribution based on binary
    })

    structure = {'binary': [], 'categorical': ['binary']}
    params = estimate_parameters(df, structure)

    # Binary: marginal probability
    assert isinstance(params['binary'][()], float)

    # Categorical: CPT with 2 entries (for binary=0 and binary=1)
    assert len(params['categorical']) == 2
    assert (0,) in params['categorical']
    assert (1,) in params['categorical']

    # Each should be a probability distribution
    for binary_val in [0, 1]:
        prob_sum = sum(params['categorical'][(binary_val,)].values())
        assert abs(prob_sum - 1.0) < 0.01

    # Compute likelihood
    log_lik = compute_log_likelihood(df, structure, params)
    assert np.isfinite(log_lik)

    print("✓ Mixed binary/categorical structure test passed")


# Continuous variable tests

def test_variable_type_detection():
    """Test that variable types are correctly detected."""
    from scripts.inference import _get_variable_type

    # Binary
    df_binary = pd.DataFrame({'x': [0, 1, 0, 1, 0]})
    assert _get_variable_type(df_binary, 'x') == 'binary'

    # Categorical
    df_categorical = pd.DataFrame({'x': [1, 2, 3, 1, 2]})
    assert _get_variable_type(df_categorical, 'x') == 'categorical'

    # Continuous
    df_continuous = pd.DataFrame({'x': [1.5, 2.3, 4.7, 3.2, 1.9]})
    assert _get_variable_type(df_continuous, 'x') == 'continuous'

    print("✓ Variable type detection test passed")


def test_continuous_no_parents():
    """Test continuous variable with no parents (Gaussian)."""
    df = pd.DataFrame({
        'X1': np.random.randn(100) * 2 + 5
    })

    structure = {'X1': []}
    params = estimate_parameters(df, structure)

    # Should be Gaussian parameters
    assert () in params['X1']
    assert 'type' in params['X1'][()]
    assert params['X1'][()]['type'] == 'gaussian'
    assert 'mean' in params['X1'][()]
    assert 'std' in params['X1'][()]

    # Mean should be close to 5, std close to 2
    assert abs(params['X1'][()]['mean'] - 5.0) < 1.0
    assert abs(params['X1'][()]['std'] - 2.0) < 1.0

    print("✓ Continuous variable (no parents) test passed")


def test_continuous_with_continuous_parents():
    """Test continuous variable with continuous parents (linear regression)."""
    np.random.seed(42)

    # Generate data: Y = 2*X1 + 3*X2 + 1 + noise
    n = 100
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    Y = 2*X1 + 3*X2 + 1 + np.random.randn(n) * 0.1

    df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

    structure = {'X1': [], 'X2': [], 'Y': ['X1', 'X2']}
    params = estimate_parameters(df, structure)

    # Should be linear regression parameters
    assert 'type' in params['Y'][()]
    assert params['Y'][()]['type'] == 'linear'
    assert 'intercept' in params['Y'][()]
    assert 'coeffs' in params['Y'][()]
    assert len(params['Y'][()]['coeffs']) == 2

    # Coefficients should be close to [2, 3], intercept close to 1
    assert abs(params['Y'][()]['intercept'] - 1.0) < 0.5
    assert abs(params['Y'][()]['coeffs'][0] - 2.0) < 0.5
    assert abs(params['Y'][()]['coeffs'][1] - 3.0) < 0.5

    print("✓ Continuous variable (with continuous parents) test passed")


def test_categorical_with_continuous_parents():
    """Test categorical variable with continuous parents (discretization)."""
    np.random.seed(42)

    # Generate data where Y depends on X
    n = 100
    X = np.random.randn(n)
    Y = (X > 0).astype(int) + 1  # Y is 1 or 2 based on sign of X

    df = pd.DataFrame({'X': X, 'Y': Y})

    structure = {'X': [], 'Y': ['X']}
    params = estimate_parameters(df, structure)

    # Should discretize continuous parent into 2 bins
    assert len(params['Y']) == 2  # Two parent combinations: X below/above median

    # Both combinations should have probability distributions over Y values
    for combo in params['Y'].values():
        assert isinstance(combo, dict)
        assert 1 in combo and 2 in combo
        assert abs(combo[1] + combo[2] - 1.0) < 0.01  # Probabilities sum to 1

    print("✓ Categorical variable (with continuous parents) test passed")


def test_continuous_likelihood():
    """Test log-likelihood computation for continuous variables."""
    np.random.seed(42)

    # Generate simple Gaussian data
    df = pd.DataFrame({'X': np.random.randn(50) * 2 + 5})

    structure = {'X': []}
    params = estimate_parameters(df, structure)
    log_lik = compute_log_likelihood(df, structure, params)

    # Log-likelihood should be finite and negative
    assert np.isfinite(log_lik)
    assert log_lik < 0

    print("✓ Continuous variable likelihood test passed")


def test_mixed_continuous_categorical():
    """Test mixed structure with continuous and categorical variables."""
    np.random.seed(42)

    # Create dataset with both continuous and categorical
    n = 100
    X1 = np.random.randn(n)  # Continuous
    X2 = (X1 > 0).astype(int)  # Binary (categorical)
    X3 = X1 * 2 + np.random.randn(n) * 0.1  # Continuous, depends on X1

    df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})

    structure = {
        'X1': [],
        'X2': ['X1'],
        'X3': ['X1']
    }

    params = estimate_parameters(df, structure)

    # X1: continuous, no parents → Gaussian
    assert params['X1'][()]['type'] == 'gaussian'

    # X2: binary, continuous parent → discretized CPT
    assert len(params['X2']) == 2  # Two bins

    # X3: continuous, continuous parent → linear regression
    assert params['X3'][()]['type'] == 'linear'

    # Compute likelihood
    log_lik = compute_log_likelihood(df, structure, params)
    assert np.isfinite(log_lik)

    print("✓ Mixed continuous/categorical structure test passed")


def test_continuous_improves_likelihood():
    """Test that adding dependencies improves likelihood for continuous data."""
    np.random.seed(42)

    # Generate data with true dependency: Y = 2*X + noise
    n = 100
    X = np.random.randn(n)
    Y = 2*X + np.random.randn(n) * 0.5

    df = pd.DataFrame({'X': X, 'Y': Y})

    # Independent structure
    structure_indep = {'X': [], 'Y': []}
    params_indep = estimate_parameters(df, structure_indep)
    log_lik_indep = compute_log_likelihood(df, structure_indep, params_indep)

    # Dependent structure
    structure_dep = {'X': [], 'Y': ['X']}
    params_dep = estimate_parameters(df, structure_dep)
    log_lik_dep = compute_log_likelihood(df, structure_dep, params_dep)

    # Dependent structure should have higher likelihood
    assert log_lik_dep > log_lik_indep
    improvement = log_lik_dep - log_lik_indep
    assert improvement > 50  # Should be substantial improvement

    print(f"✓ Continuous dependency test passed (improvement: {improvement:.2f})")


if __name__ == "__main__":
    print("Running tests for data generation and DAG enumeration...\n")

    # Data generation tests
    print("=== Data Generation Tests ===")
    test_chain_data_shape()
    test_chain_data_columns()
    test_chain_data_binary()
    test_chain_data_deterministic()
    test_chain_structure()

    test_fork_data_shape()
    test_fork_data_columns()
    test_fork_data_binary()
    test_fork_data_deterministic()

    test_collider_data_shape()
    test_collider_data_columns()
    test_collider_data_binary()
    test_collider_data_deterministic()
    test_collider_independence()

    # DAG enumeration tests
    print("\n=== DAG Enumeration Tests ===")
    test_generate_all_dags_count()
    test_generate_all_dags_structure()
    test_generate_all_dags_acyclic()
    test_count_edges()
    test_is_dag_valid()
    test_is_dag_cycle()
    test_known_structures_present()

    # Bayesian inference tests
    print("\n=== Bayesian Inference Tests ===")
    test_estimate_parameters_marginal()
    test_estimate_parameters_conditional()
    test_compute_log_likelihood()
    test_compute_posteriors_format()
    test_compute_posteriors_true_structure_wins()
    test_compute_posteriors_penalty()

    # Probabilistic query tests
    print("\n=== Probabilistic Query Tests ===")
    test_compute_joint_prob()
    test_query_single_structure_simple()
    test_query_single_structure_marginalization()
    test_query_with_uncertainty()
    test_query_with_uncertainty_chain_data()

    # Stochastic search tests
    print("\n=== Stochastic Search Tests ===")
    test_score_theory_empty()
    test_score_theory_single_parent()
    test_sample_theories()
    test_stochastic_beam_search_chain()
    test_stochastic_beam_search_fork()

    # Categorical variable tests
    print("\n=== Categorical Variable Tests ===")
    test_categorical_no_parents()
    test_categorical_with_categorical_parents()
    test_categorical_likelihood()
    test_mixed_binary_categorical()

    # Continuous variable tests
    print("\n=== Continuous Variable Tests ===")
    test_variable_type_detection()
    test_continuous_no_parents()
    test_continuous_with_continuous_parents()
    test_categorical_with_continuous_parents()
    test_continuous_likelihood()
    test_mixed_continuous_categorical()
    test_continuous_improves_likelihood()

    print("\n✅ All tests passed!")
