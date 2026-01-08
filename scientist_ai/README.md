# Scientist AI - Rust Implementation

Bayesian causal structure learning implementation in Rust.

## Status

**Implementation:** Complete ✓
**Validation:** Phase 1-3 complete, quality experiments planned
**Performance:** 297 structures/second on Iris (5 vars, 150 samples)

## Overview

Clean implementation of Scientist AI algorithm in Rust, validated against Python/Julia reference implementations.

### Key Features
- **Binary variables**: CPTs with Laplace smoothing, multinomial likelihood
- **Categorical variables**: K-value multinomial distributions
- **Continuous variables**: Gaussian (no parents) or linear regression (with parents)
- **Mixed networks**: Median discretization for continuous→categorical edges
- **Scoring methods**: Edge-based (λ penalty), BIC, BDeu (discrete only)
- **Search**: Stochastic beam search with softmax sampling

## Project Structure

```
src/
├── main.rs         # CLI entry point, TOML config parsing
├── dag.rs          # DAG structures, cycle detection
├── inference.rs    # Bayesian scoring (2,130 lines)
├── search.rs       # Beam search + exhaustive enumeration (382 lines)
└── bin/
    ├── exhaustive_search.rs  # Enumerate all DAGs (small datasets)
    └── profile_scoring.rs    # Performance profiling
```

## Implementation Checklist

### Core Implementation ✓
- [x] Module structure (dag, inference, search)
- [x] TOML config parsing with flexible scoring methods
- [x] DAG cycle detection (DFS-based)
- [x] Phase 1: Binary variables (type detection, CPT, log-likelihood)
- [x] Phase 2: Categorical variables (multinomial distributions)
- [x] Phase 3: Continuous variables (Gaussian, linear regression)
- [x] Phase 4: Mixed networks (median discretization)
- [x] Phase 5: Multi-structure scoring (compute_posteriors, log-sum-exp normalization)
- [x] Stochastic beam search (generate_neighbors, sample_structures)
- [x] Exhaustive DAG enumeration (for validation)
- [x] Three scoring methods (EdgeBased, BIC, BDeu)

### Bug Fixes ✓
- [x] Fixed infinite loop in sample_structures (renormalize after exclusion)
- [x] Fixed string categorical detection for Species column
- [x] Fixed type casting (i64→f64) for categorical variables in regression
- [x] Implemented mixed network log-likelihood for edge-based/BIC scoring

### Tests ✓
- [x] 10 inference tests (binary, categorical, continuous, mixed, scoring methods)
- [x] 3 search tests (neighbor generation, beam search, exhaustive enumeration)
- [x] All tests passing

## Validation Status

### Phase 1: Score Validation (Single Structure) ✓
**Goal:** Verify scoring math is correct
**Method:** Score Julia MAP structure on Iris dataset
**Result:** Rust -350.892 vs Julia -350.89 (diff: 0.002) ✓

### Phase 2: Exhaustive Search Validation (Complete Distribution) ✓
**Goal:** Verify posterior distribution computation is correct
**Method:** Exhaustive enumeration on 3-variable chain (all 25 DAGs)
**Result:** All posteriors match Python within 5.6e-15 (machine precision) ✓
**Key finding:** Top 3 structures (equivalent chains) share 98.7% posterior mass

### Phase 3: Beam Search Validation (Algorithm Correctness) ✓
**Goal:** Verify beam search finds optimal structures
**Method:** Beam search on 3-variable chain, compare to exhaustive results
**Result:** Found structure X3→X2→X1 with score -176.154 (matches exhaustive top-3) ✓
**Convergence:** Round 2 (no improvement after)

### Phase 4: Quality Validation (Planned)

**Experiment 1: Ground Truth Comparison** [PRIORITY 1]
- **Dataset:** 3-4 variable subsets of Iris/SECOM
- **Method:** Exhaustive search (optimal) vs beam search (20 runs, different seeds)
- **Metrics:** Success rate (% finding optimal), score gap (mean vs optimal)
- **Claim:** "Beam search finds optimal structures X% of the time"

**Experiment 2: Baseline Comparison** [PRIORITY 3]
- **Dataset:** Full Iris (5 vars), SECOM subsets (10+ vars)
- **Method:** Beam search vs random structure generation baseline
- **Metrics:** Score distributions, effect size (Cohen's d or similar)
- **Claim:** "Beam search finds structures Y std devs better than random"

**Experiment 3: Cross-Implementation Quality** [PRIORITY 2]
- **Dataset:** Iris, SECOM
- **Method:** Rust vs Python beam search (20 runs each, different seeds)
- **Metrics:** Mean score, std dev, t-test for difference
- **Claim:** "Rust achieves comparable quality to Python reference"

**Experiment 4: Convergence Analysis** [PRIORITY 4]
- **Method:** Score vs iteration plots for multiple runs
- **Metrics:** Convergence round, score trajectory
- **Claim:** "Beam search converges within X rounds"

## Running Experiments

### Basic Usage

Create a TOML config file:

```toml
# experiments/my_experiment.toml
input = "data/iris_numeric.csv"
output = "results/my_results.json"
beam_size = 10
max_rounds = 20

[[scoring]]
method = "BDeu"
alpha = 1.0
```

Run:
```bash
cargo run --release --bin scientist_ai experiments/my_experiment.toml
```

### Example Configs

**Single scoring method:**
```toml
[[scoring]]
method = "EdgeBased"
lambda = 2.0
```

**Multiple methods:**
```toml
[[scoring]]
method = "BDeu"
alpha = 1.0

[[scoring]]
method = "BIC"

[[scoring]]
method = "EdgeBased"
lambda = 0.5
```

### Exhaustive Search (Small Datasets)

For ≤5 variables, enumerate all possible DAGs:

```bash
cargo run --release --bin exhaustive_search
# Uses data/chain_3var.csv by default
# Outputs to results/exhaustive_rust.json
```

### Performance Profiling

```bash
cargo run --release --bin profile_scoring
# Times scoring on Iris dataset
# Reports structures/second
```

## Performance

**Iris Dataset (5 vars, 150 samples):**
- Scoring: 297 structures/second (~3.4ms per structure)
- Beam search: 5 rounds with beam=5 completes in seconds
- Bottleneck: Mixed network scoring (continuous + categorical variables)

**3-Variable Chain (100 samples):**
- Exhaustive: All 25 DAGs scored instantly
- Beam search: Converges in 2 rounds

## Dependencies

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"
polars = { version = "0.45", features = ["lazy", "csv"] }
itertools = "0.13"
statrs = "0.17"  # Gamma function for BDeu
rand = "0.8"     # Random sampling in search
```

## Data Format

CSV files with:
- Header row with variable names
- Binary variables: 0/1 integer values
- Categorical variables: Integer codes (≤10 unique values)
- Continuous variables: Float values

**Example:**
```csv
SepalLength,SepalWidth,Species
5.1,3.5,1
4.9,3.0,1
7.0,3.2,3
```

## Output Format

JSON with structure results:

```json
{
  "config": {
    "input": "data/iris_numeric.csv",
    "beam_size": 10,
    "max_rounds": 20
  },
  "data": {
    "rows": 150,
    "columns": 5,
    "variables": ["SepalLength", "SepalWidth", ...]
  },
  "methods": [
    {
      "method": "BDeu (α=1)",
      "best_structure": {
        "SepalLength": ["SepalWidth"],
        "Species": []
      },
      "edges": 1,
      "score": -350.892
    }
  ]
}
```

## Known Limitations

1. **BDeu scoring:** Only works for discrete (binary/categorical) variables. Use BIC or EdgeBased for mixed networks.
2. **Continuous parents of categorical children:** Uses median discretization (simple but loses information). Future: one-hot encoding or ANCOVA.
3. **Performance:** Mixed network scoring not optimized. Potential improvements:
   - Cache variable types (currently recomputed per structure)
   - Cache parameter estimates for common parent sets
   - Batch linear regression operations

## Next Steps

1. Run Experiment 1 (beam search vs exhaustive on 3-4 vars) for quality validation
2. Run Experiment 3 (cross-implementation comparison) for correctness validation
3. Document validation results in research log
4. Performance optimization if needed for larger experiments
5. Medium-scale experiments (SECOM, 10-100 variables) for Paper 1 data collection

## References

- Python reference: `~/Documents/research/code/python/scripts/`
- Validation data: `data/chain_3var.csv`, `data/iris_numeric.csv`
- Results: `results/exhaustive_*.json`, `results/*_beam_search.json`
