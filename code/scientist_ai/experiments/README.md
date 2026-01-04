# Experiment Configuration

This folder contains TOML configuration files for running structure learning experiments.

## Configuration Format

### Basic Fields

```toml
input = "data/train.csv"    # Path to input data
output = "results/run1.json" # Path to output results
beam_size = 10               # Beam search width
max_rounds = 3               # Maximum search iterations
```

### Scoring Methods

#### Single Method

```toml
scoring = { method = "edge", lambda = 2.0 }
```

#### Multiple Methods (for comparison)

```toml
scoring = [
    { method = "edge", lambda = 2.0 },
    { method = "bic" },
    { method = "bdeu", alpha = 1.0 }
]
```

### Available Scoring Methods

| Method | Parameters | Example | Use Case |
|--------|------------|---------|----------|
| `edge` | `lambda` (default: 2.0) | `{ method = "edge", lambda = 2.0 }` | Fast, tunable, works for all variable types |
| `bic` | None | `{ method = "bic" }` | Theoretically grounded, automatic penalty |
| `bdeu` | `alpha` (default: 1.0) | `{ method = "bdeu", alpha = 1.0 }` | Gold standard for discrete variables |

### Examples

See the example files in this folder:
- `example1_single_method.toml` - Single scoring method
- `example2_multiple_methods.toml` - Compare three methods
- `example3_lambda_tuning.toml` - Tune lambda parameter
- `example4_legacy.toml` - Legacy format (backwards compatible)

### Running an Experiment

```bash
cargo run experiments/example1_single_method.toml
```

### Legacy Format (Deprecated)

Old format using just `lambda` is still supported:

```toml
lambda = 3.0  # Converted to EdgeBased(3.0)
```

But the new `scoring` format is preferred for flexibility.
