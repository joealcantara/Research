"""
Utility functions for displaying and analyzing results.

Functions for formatting output, printing posteriors, and analyzing
the results of Bayesian causal structure learning.
"""


def format_structure(structure):
    """
    Format a DAG structure as a readable string.

    Args:
        structure: DAG structure dict, e.g., {'X1': [], 'X2': ['X1'], 'X3': ['X2']}

    Returns:
        Readable string representation of the structure
    """
    edges = []
    for var, parents in structure.items():
        for parent in parents:
            edges.append(f"{parent}→{var}")

    if len(edges) == 0:
        return "(independent)"
    else:
        return ", ".join(edges)


def get_top_structures(results, n=10):
    """
    Get the top N structures by posterior probability.

    Args:
        results: Results dict from compute_posteriors()
        n: Number of top structures to return

    Returns:
        List of (name, result) tuples sorted by posterior (highest first)
    """
    sorted_results = sorted(results.items(), key=lambda x: -x[1]['posterior'])
    return sorted_results[:n]


def print_posteriors(results, top_n=None):
    """
    Pretty print posterior results.

    Args:
        results: Results dict from compute_posteriors()
        top_n: If specified, only print top N structures
    """
    print(f"{'Structure':30} | {'Edges':5} | {'Log-lik':>10} | {'Log-prior':>10} | {'Log-post':>10} | {'Posterior':>10}")
    print("-" * 100)

    # Sort by posterior
    sorted_results = sorted(results.items(), key=lambda x: -x[1]['posterior'])

    # Limit to top_n if specified
    if top_n is not None:
        sorted_results = sorted_results[:top_n]

    for name, r in sorted_results:
        # Handle both string names and integer indices
        name_str = str(name) if not isinstance(name, str) else name
        print(f"{name_str:30} | {r['edges']:5} | {r['log_likelihood']:10.2f} | {r['log_prior']:10.2f} | {r['log_posterior']:10.2f} | {r['posterior']:10.4f}")


def print_posterior_summary(results):
    """
    Print summary statistics of the posterior distribution.

    Args:
        results: Results dict from compute_posteriors()
    """
    sorted_results = get_top_structures(results, n=len(results))

    print("Posterior concentration:")
    print(f"  Top 1:   {sorted_results[0][1]['posterior']:.6f}")

    if len(sorted_results) >= 5:
        top_5_prob = sum(r[1]['posterior'] for r in sorted_results[:5])
        print(f"  Top 5:   {top_5_prob:.6f}")

    if len(sorted_results) >= 10:
        top_10_prob = sum(r[1]['posterior'] for r in sorted_results[:10])
        print(f"  Top 10:  {top_10_prob:.6f}")

    if len(sorted_results) >= 20:
        top_20_prob = sum(r[1]['posterior'] for r in sorted_results[:20])
        print(f"  Top 20:  {top_20_prob:.6f}")

    if len(sorted_results) >= 50:
        top_50_prob = sum(r[1]['posterior'] for r in sorted_results[:50])
        print(f"  Top 50:  {top_50_prob:.6f}")

    if len(sorted_results) >= 100:
        top_100_prob = sum(r[1]['posterior'] for r in sorted_results[:100])
        print(f"  Top 100: {top_100_prob:.6f}")


def print_query_result(query_var, query_val, evidence, weighted_answer, breakdown, top_n=5):
    """
    Pretty print probabilistic query results.

    Args:
        query_var: Variable queried
        query_val: Value queried
        evidence: Evidence dict
        weighted_answer: Final weighted probability
        breakdown: Per-structure breakdown from query_with_uncertainty()
        top_n: Number of top contributing structures to show
    """
    evidence_str = ', '.join([f"{k}={v}" for k, v in evidence.items()])
    print(f"Query: P({query_var}={query_val} | {evidence_str})")
    print(f"Weighted answer: {weighted_answer:.4f}")
    print()
    print(f"{'Structure':30} | {'P(answer)':>10} | {'Posterior':>10} | {'Contribution':>12}")
    print("-" * 70)

    # Sort by posterior and show top N
    sorted_breakdown = sorted(breakdown.items(), key=lambda x: -x[1]['posterior'])
    for name, r in sorted_breakdown[:top_n]:
        name_str = str(name) if not isinstance(name, str) else name
        print(f"{name_str:30} | {r['answer']:10.4f} | {r['posterior']:10.4f} | {r['contribution']:12.4f}")
