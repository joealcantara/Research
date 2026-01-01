# Scientist AI v0.1

Bayesian causal structure learning with support for binary, categorical, and continuous variables.

This project implements core algorithms for automated causal discovery from the Scientist AI paper, with extensions to support multiple variable types and theory languages.

## Project Structure

```
scientist_ai_v0.1/
├── scripts/
│   ├── generate_data.py    # Synthetic data generation (chain, fork, collider)
│   ├── dag_utils.py         # DAG structure utilities and enumeration
│   ├── inference.py         # Bayesian inference (binary, categorical, continuous)
│   ├── search.py            # Stochastic search algorithms (beam search)
│   └── utils.py             # Display and formatting utilities
├── notebooks/               # Jupyter notebooks for exploration
├── tests.py                 # Comprehensive test suite (48 tests)
├── demo_3var_workflow.py    # Demo: complete 3-variable workflow
├── iris_benchmark.py        # Iris benchmark with binary discretization
├── iris_continuous_benchmark.py  # Iris benchmark with continuous variables
├── iris_mixed_theory.py     # Iris benchmark with mixed theory assignment
├── iris_joint_search.py     # Joint search over (structure, theory) pairs
├── wine_beam_search.py      # Wine dataset (14 vars) with beam search
├── theory_comparison.py     # Compare linear vs decision tree theories per variable
├── test_continuous.py       # Tests for continuous variable support
├── pyproject.toml           # Project dependencies (uv)
└── README.md                # This file
```

## Features

### Variable Types Supported

The inference engine handles three variable types:

1. **Binary** (0/1): Traditional binary variables
   - Model: Conditional probability tables (CPTs)

2. **Categorical** (1, 2, 3, ...): Discrete multi-valued variables
   - Model: Conditional probability distributions
   - Example: Species (setosa, versicolor, virginica)

3. **Continuous** (real-valued): Continuous measurements
   - Model: Linear Gaussian (linear regression + Gaussian noise)
   - Example: Sepal length, petal width

**Key insight**: Continuous variables provide dramatically better structure identification than binary discretization (5.3x improvement on Iris dataset).

### Data Generation (`scripts/generate_data.py`)
Generate synthetic data from known causal structures for testing and validation:
- **Chain structure** (X1 → X2 → X3): Sequential dependencies
- **Fork structure** (X2 ← X1 → X3): Common cause
- **Collider structure** (X1 → X2 ← X3): Common effect

```python
from scripts.generate_data import generate_chain_data

# Generate 100 samples from a chain structure
df = generate_chain_data(n_samples=100, seed=42)
```

### DAG Utilities (`scripts/dag_utils.py`)
Tools for working with directed acyclic graphs:
- **`generate_all_dags(variables)`**: Enumerate all possible DAG structures
  - 3 variables → 25 DAGs
  - 5 variables → 29,281 DAGs
- **`is_dag(edges, n)`**: Check if edges form a valid DAG (cycle detection)
- **`count_edges(structure)`**: Count edges in a DAG structure

```python
from scripts.dag_utils import generate_all_dags

# Generate all 25 possible DAGs for 3 variables
variables = ['X1', 'X2', 'X3']
all_dags = generate_all_dags(variables)
print(f"Generated {len(all_dags)} DAGs")  # 25
```

### Bayesian Inference (`scripts/inference.py`)
Core algorithms for causal structure learning with support for binary, categorical, and continuous variables.

**Structure Learning:**
- **`estimate_parameters(df, structure)`**: Learn parameters from data
  - Binary/categorical: Conditional probability tables
  - Continuous: Linear Gaussian models (regression + noise)
- **`compute_log_likelihood(df, structure, params)`**: Score how well a structure explains the data
- **`compute_posteriors(df, structures, lambda_penalty)`**: Compute posterior probability for each structure

**Probabilistic Queries:**
- **`query_single_structure(query_var, query_val, evidence, structure, params)`**: Compute P(query | evidence) for one structure
- **`query_with_uncertainty(query_var, query_val, evidence, structures, results)`**: Bayesian model averaging across all structures

```python
from scripts.generate_data import generate_chain_data
from scripts.dag_utils import generate_all_dags
from scripts.inference import compute_posteriors

# Generate data from chain structure
df = generate_chain_data(200, seed=42)

# Generate all possible structures
variables = ['X1', 'X2', 'X3']
all_structures = generate_all_dags(variables)

# Score each structure against the data
results = compute_posteriors(df, all_structures, lambda_penalty=2.0)

# Find the best structure
best = max(results.items(), key=lambda x: x[1]['posterior'])
print(f"Best structure: {best[0]} with posterior {best[1]['posterior']:.4f}")
```

### Stochastic Search (`scripts/search.py`)
Efficient search algorithms for large-scale structure learning:

When enumerating all DAGs becomes intractable (n > 5 variables), use beam search:
- **`score_theory(df, parent_vars, target, lambda_penalty)`**: Score a theory (parents → target)
- **`sample_theories(scored_theories, k, temperature)`**: Sample theories proportionally to scores
- **`stochastic_beam_search(df, target, k, max_rounds, ...)`**: Find high-scoring parent sets without enumeration

**Scalability:**
- n=3: 25 DAGs → enumeration works fine
- n=5: 29,281 DAGs → enumeration slow but feasible
- n=6: 3.7M DAGs → beam search necessary
- n=10: billions of DAGs → only beam search is practical

```python
from scripts.generate_data import generate_chain_data
from scripts.search import stochastic_beam_search

# Generate data
df = generate_chain_data(100, seed=42)

# Search for theories explaining X3 (much faster than enumerating all DAGs)
results = stochastic_beam_search(df, target='X3', k=10, max_rounds=3)

# Show best theories
for theory, score in results[:3]:
    parents = ' + '.join(theory) if theory else '(nothing)'
    print(f"{parents} → X3 | score: {score:.2f}")
```

### Display Utilities (`scripts/utils.py`)
Helper functions for formatting and displaying results:
- **`print_posteriors(results, top_n)`**: Pretty print posterior results
- **`print_posterior_summary(results)`**: Show posterior concentration statistics
- **`print_query_result(...)`**: Display probabilistic query results
- **`format_structure(structure)`**: Format DAG as readable string

## Installation

This project uses `uv` for dependency management:

```bash
cd scientist_ai_v0.1
uv sync
```

**Dependencies:**
- numpy
- pandas
- matplotlib
- scikit-learn
- pytest

## Running Tests

The project includes 48 comprehensive tests covering all functionality:

```bash
# Run all tests
uv run python tests.py

# Or with pytest for verbose output
uv run pytest tests.py -v
```

**Test Coverage:**
- ✓ 14 Data generation tests (shape, columns, binary values, deterministic, statistical structure)
- ✓ 7 DAG enumeration tests (count, format, acyclicity, edge counting, cycle detection)
- ✓ 6 Bayesian inference tests (parameter estimation, likelihood, posteriors, penalties)
- ✓ 5 Probabilistic query tests (joint probabilities, marginalization, model averaging)
- ✓ 5 Stochastic search tests (theory scoring, sampling, beam search on chain/fork data)
- ✓ 4 Categorical variable tests (multi-valued discrete variables)
- ✓ 7 Continuous variable tests (Linear Gaussian models, mixed structures)

## Benchmarks

### Iris Dataset: Theory Language Comparison

Demonstrates the importance of theory language expressiveness using the classic Iris dataset (150 samples, 5 variables).

**Run benchmarks:**
```bash
# Binary discretization (baseline)
uv run python iris_benchmark.py

# Continuous variables (Linear Gaussian)
uv run python iris_continuous_benchmark.py

# Mixed theory assignment (variable-specific theories)
uv run python iris_mixed_theory.py
```

**Results:**

| Metric | Binary | Linear Gaussian | Mixed Theory | Best Improvement |
|--------|--------|----------------|--------------|------------------|
| Top 1 posterior | 3.3% | 17.5% | **35.1%** | **10.6x over binary** |
| Top 10 cumulative | 23.5% | 76.3% | **95.7%** | **4.1x over binary** |
| Top 100 cumulative | 60.2% | 100.0% | **100.0%** | **1.7x over binary** |

**Key findings:**
1. **Mixed theory assignment doubles structure identification** (35.1% vs 17.5% for uniform Linear Gaussian)
2. **Variable-specific theory languages beat uniform approaches** - Using decision trees for categorical variables (species) and linear regression for continuous relationships achieves 2.01x improvement over pure Linear Gaussian
3. **Theory choice affects structure identification** - Mixed theory identifies a different best structure than Linear Gaussian, showing that theory language influences which causal relationships are discovered
4. **Strong posterior concentration** - 95.7% probability in just 10 structures (vs 76.3% for Linear Gaussian)

**Why this matters:**
- Binary discretization destroys information: "sepal length > median" loses magnitude
- Linear Gaussian preserves continuous relationships: "sepal_length = 2.5 × petal_length + 1.2"
- Mixed theory uses the right language for each variable: decision trees for classification (species), linear for continuous relationships (measurements)
- Better theory language → better causal discovery

This validates the **Scientist AI paper's core thesis**: the expressiveness and appropriateness of your theory language fundamentally determines how well you can identify causal structure.

### Wine Dataset: Scaling to 14 Variables with Beam Search

Tests theory language comparison on a larger dataset (178 samples, 14 variables) where exhaustive enumeration is intractable.

**Run benchmark:**
```bash
# Beam search with Linear Gaussian, Decision Trees, and Mixed Theory
uv run python wine_beam_search.py
```

**Results:**

| Approach | Total Score | Improvement |
|----------|-------------|-------------|
| Linear Gaussian | -3044.53 | baseline |
| **Decision Trees (depth=2)** | **-2977.51** | **+67.02** ✓ |
| Mixed Theory | -3032.46 | +12.07 |

**Key findings:**
1. **Pure decision trees won!** (opposite to Iris where mixed theory won) - With 14 chemical composition variables, non-linear relationships dominate
2. **Dataset-dependent results** - Morphological data (Iris) prefers matched theories (trees for categorical, linear for continuous), but chemical data (Wine) shows pervasive non-linearity where uniform trees outperform
3. **Beam search scales effectively** - With 14 variables, billions of possible structures make enumeration impossible; beam search found good structures efficiently
4. **Theory language choice is domain-specific** - The optimal strategy depends on the nature of relationships in the data, not just variable types

**Why this matters:**
- Wine's 13 chemical features (alcohol, phenols, flavanoids, etc.) have complex non-linear interactions
- Decision trees (even shallow depth=2) capture these better than linear regression
- Mixed theory only helped marginally (+12.07) because continuous variables also benefit from trees here
- Validates that theory language selection requires understanding the domain, not just following type-based rules

**Comparison to Iris:**
- **Iris (morphological):** Simple linear relationships between measurements → mixed theory wins (35.1%)
- **Wine (chemical):** Complex non-linear interactions between compounds → decision trees win (-2977.51)
- **General lesson:** Theory language effectiveness is dataset-dependent; no universal "best" approach

This extends the Scientist AI thesis: theory language matters, AND the optimal choice depends on domain characteristics beyond variable types.

### Theory Language Exploration: Which Variables Need Logical Rules?

Going beyond uniform theory languages, we can analyze **which specific variables benefit from which theory types**.

**Run analysis:**
```bash
uv run python theory_comparison.py
```

**Key findings from Iris dataset:**

Starting from the best Linear Gaussian structure (17.5% posterior), we compared linear vs decision tree models for each variable:

| Variable | Linear Score | Tree Score | Improvement | Conclusion |
|----------|-------------|------------|-------------|------------|
| **species** | -137.76 | -123.53 | **+14.23** | ✓ Trees win (logical rules) |
| **petal_width** | +25.10 | +27.71 | **+2.61** | ✓ Trees win (marginal) |
| **petal_length** | -31.48 | -84.05 | **-52.57** | ✗ Linear much better |
| **sepal_length** | -100.98 | -106.54 | **-5.56** | ✗ Linear better |

**Insight: Different variables need different theory languages!**

- **Categorical targets** (species) → Decision trees capture IF-THEN rules: "IF petal_width < 0.8 THEN setosa"
- **Continuous relationships** (petal_length from other measurements) → Linear regression captures proportional relationships

This analysis informed the **mixed theory assignment** approach (see results above), which achieved 2.01x improvement over uniform Linear Gaussian by using variable-specific theory languages.

**Completed explorations:**
1. ✓ **Mixed theory assignment** (Option 1): Achieved 35.1% top posterior (2.01x improvement on Iris)
2. ✓ **Joint search** (Option 2): Implemented but greedy sequential optimization (35.1%) outperformed local joint search (26.8%) with limited compute budget
3. ✓ **Scaling validation**: Wine dataset (14 variables) confirms theory language matters, shows domain-dependent optimal strategies

**Future directions:**
1. **Decision tree depth tuning**: Optimize max_depth per variable for complexity/accuracy tradeoff
2. **Additional datasets**: Test on more domains to characterize when linear/tree/mixed is optimal
3. **Richer theory languages**: Beyond linear/tree - polynomial, splines, neural networks

This implementation validates the Scientist AI vision: automatically discovering both **causal structure** AND **appropriate theory languages** for each relationship significantly improves causal discovery.

## Usage Examples

### Example 1: Structure Learning on Synthetic Data

```python
from scripts.generate_data import generate_chain_data
from scripts.dag_utils import generate_all_dags
from scripts.inference import compute_posteriors
from scripts.utils import print_posteriors

# Generate chain data
df = generate_chain_data(100, seed=42)

# Define candidate structures
structures = {
    'chain': {'X1': [], 'X2': ['X1'], 'X3': ['X2']},
    'fork': {'X1': [], 'X2': ['X1'], 'X3': ['X1']},
    'collider': {'X1': [], 'X2': ['X1', 'X3'], 'X3': []},
    'independent': {'X1': [], 'X2': [], 'X3': []}
}

# Compute posteriors
results = compute_posteriors(df, structures, lambda_penalty=2.0)

# Display results
print_posteriors(results)
```

### Example 2: Probabilistic Queries with Uncertainty

```python
from scripts.generate_data import generate_chain_data
from scripts.dag_utils import generate_all_dags
from scripts.inference import compute_posteriors, query_with_uncertainty
from scripts.utils import print_query_result

# Generate data and learn structures
df = generate_chain_data(200, seed=42)
structures = {
    'chain': {'X1': [], 'X2': ['X1'], 'X3': ['X2']},
    'fork': {'X1': [], 'X2': ['X1'], 'X3': ['X1']},
}

# Compute posteriors (with parameters stored for queries)
results = compute_posteriors(df, structures, lambda_penalty=2.0, store_params=True)

# Query: What is P(X3=1 | X1=1)?
answer, breakdown = query_with_uncertainty('X3', 1, {'X1': 1}, structures, results)

# Display results
print_query_result('X3', 1, {'X1': 1}, answer, breakdown)
```

### Example 3: Continuous Variables on Real Data

```python
from sklearn.datasets import load_iris
import pandas as pd
from scripts.dag_utils import generate_all_dags
from scripts.inference import compute_posteriors

# Load Iris dataset (keep continuous features)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width',
                                       'petal_length', 'petal_width'])
df['species'] = iris.target + 1  # Categorical: 1, 2, 3

# Generate all possible structures
all_dags = generate_all_dags(df.columns.tolist())

# Learn structure (uses Linear Gaussian for continuous, CPT for categorical)
results = compute_posteriors(df, all_dags, lambda_penalty=2.0)

# Best structure has 17.5% posterior (vs 3.3% with binary discretization!)
best = max(results.items(), key=lambda x: x[1]['posterior'])
print(f"Best structure: {best[1]['posterior']:.1%} posterior")
```

## Key Insights Validated

This implementation validates several key theoretical results:

1. **Markov Equivalence**: Chain and fork structures are hard to distinguish from observational data alone
2. **Collider Identification**: Collider structures are identifiable (X1 and X3 independent unless conditioning on X2)
3. **Bayesian Model Averaging**: Accounting for structural uncertainty improves predictions
4. **Complexity Penalty**: Simpler structures are favored with appropriate penalization
5. **Theory Language Matters**: Continuous models (5.3x better) >> Binary discretization
6. **Variable-Specific Theory Languages**: Different variables need different theory types
   - Categorical targets benefit from decision trees (logical IF-THEN rules)
   - Continuous relationships prefer linear regression (proportional relationships)
   - Mixed theory assignments (35.1% top posterior on Iris) outperform both uniform Linear Gaussian (17.5%) and uniform decision trees
7. **Theory Language Affects Structure Discovery**: Using appropriate theories changes which causal structure is identified as most likely - mixed theory found a different MAP structure than Linear Gaussian
8. **Domain-Dependent Theory Selection**: Optimal theory language depends on data characteristics, not just variable types
   - Morphological data (Iris): Mixed theory wins (35.1%) - simple linear relationships
   - Chemical data (Wine): Decision trees win (-2977.51) - complex non-linear interactions
   - No universal "best" approach - must match theory expressiveness to domain complexity
9. **Beam Search Scalability**: Stochastic search enables theory language comparison on larger datasets (14+ variables) where exhaustive enumeration is intractable

## Development Workflow

**Test-Driven Development:**
1. Generate synthetic data with known ground truth
2. Test algorithms recover the correct structure
3. Validate on real data (e.g., Iris dataset)
4. Add tests to prevent regressions

**Adding New Features:**
1. Write function in appropriate script (`scripts/`)
2. Add comprehensive tests (`tests.py`)
3. Run test suite to verify correctness
4. Document in README

## Future Work

- **Joint search optimization** (Option 2): Beam search over (structure, theory assignment) pairs for full joint optimization
- **Greedy theory upgrade** (Option 4): Start simple, incrementally upgrade theories for variables that benefit most
- **Decision tree depth tuning**: Optimize max_depth per variable for complexity/accuracy tradeoff
- **Decision tree Bayesian networks**: Full tree-based theory languages (reproduce Dec 30 result: 32.56% top posterior)
- **Scaling to larger graphs**: Optimize beam search for n > 10 variables
- **More benchmarks**: UCI datasets, synthetic benchmarks
- **Visualization**: Plot DAG structures and posterior distributions
- **Interventional queries**: Extend to do(X) queries and counterfactuals

## References

- Scientist AI paper: Automated causal discovery with theory languages
- Pearl, J. (2009). Causality: Models, Reasoning and Inference
- Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models

## License

Educational/Research use
