# Scientist AI - Rust Implementation

Bayesian causal structure learning implementation in Rust.

## Overview

This is a clean rebuild of the Scientist AI algorithm, consolidating previous Python/Julia implementations into a single, fast, maintainable Rust codebase.

**Speed:** Rust shows 4.5-6.6x performance advantage over Julia on 100-variable benchmarks.

**Reference Implementation:** Python code at `~/Documents/research/code/python/scripts/` provides the algorithmic specification.

## Project Structure

```
src/
├── main.rs         # CLI entry point, TOML config parsing
├── dag.rs          # DAG structures, cycle detection
├── inference.rs    # Bayesian scoring, parameter estimation
└── search.rs       # Stochastic beam search algorithm
```

## Build Sequence

- [x] Initialize Rust project with cargo
- [ ] Set up module structure (dag.rs, inference.rs, search.rs)
- [ ] Implement TOML config file parsing for experiment parameters
- [ ] Implement DAG data structures and cycle detection (reference: `dag_utils.py`)
- [ ] Implement Bayesian inference engine (reference: `inference.py`)
- [ ] Implement structure learning search algorithm (reference: `search.py`)
- [ ] Add test cases with known structures (chain, fork, collider)
- [ ] Validate: Rust outputs match Python outputs on test cases
- [ ] Retire Python/Julia implementations once validated

## Module Specifications

### dag.rs
Based on `python/scripts/dag_utils.py`:
- DAG structure representation (`HashMap<String, Vec<String>>` for `{var: [parents]}`)
- `is_dag()` - cycle detection via depth-first search
- `count_edges()` - utility function

### inference.rs
Based on `python/scripts/inference.py`:
- `estimate_parameters()` - learn conditional probabilities from data
- `score_theory()` - compute log posterior (likelihood + complexity penalty)
- Support binary, categorical, and continuous variables

### search.rs
Based on `python/scripts/search.py`:
- `stochastic_beam_search()` - efficient structure learning for large variable sets
- `sample_theories()` - softmax sampling for exploration
- Handles: beam width, max rounds, temperature, lambda penalty

## Running Experiments

Create experiment config files:

**experiments/secom_baseline.toml:**
```toml
input = "data/secom.csv"
beam_size = 10
max_rounds = 3
lambda = 2.0
```

Run:
```bash
cargo run --release -- experiments/secom_baseline.toml
```

## Validation Strategy

1. Generate test data with Python (`generate_data.py`: chain, fork, collider)
2. Run both Python and Rust on same CSV files
3. Compare outputs (structures, scores, parameters)
4. Once validated → retire Python/Julia for experiments
5. Keep Python for data analysis/visualization only

## Dependencies

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
csv = "1.3"
```

Add more as needed (e.g., ndarray for matrix operations).

## Notes

- Python implementation remains as reference specification
- This is the production implementation for all PhD experiments
- Rust edition 2024 (latest language features)
