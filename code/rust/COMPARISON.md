# Python vs Julia vs Rust Performance Comparison

## Iris Benchmark (29,281 DAG structures)

### Performance Results

| Implementation | Total Time | DAG Generation | Inference | Speedup vs Python |
|----------------|-----------|----------------|-----------|-------------------|
| **Python** | 7:53 (473s) | ~1s | ~472s | 1.0x (baseline) |
| **Julia** | 1:18 (78s) | 1.6s | 69s | **6.1x faster** |
| **Rust** | 0.88s | 0.16s | 0.23s | **537x faster** |

### Results Validation

❌ **PROBLEM: Rust results DO NOT match Python/Julia**

**Top 1 Posterior:**
- Python: 0.174838 (Structure 28708)
- Julia: 0.174838 (Structure 28709)  ✓ Match
- Rust: 0.132580 (Structure 25441)   ❌ Different!

**MAP Structure:**

Python/Julia (identical):
```
sepal_length: ← sepal_width, species
sepal_width: (no parents - root node)
petal_length: ← sepal_length, sepal_width, petal_width, species
petal_width: ← sepal_length, species
species: ← sepal_width
```

Rust (WRONG):
```
sepal_length: ← petal_width
sepal_width: ← sepal_length, species
petal_length: ← sepal_length, sepal_width, petal_width, species
petal_width: ← species
species: (no parents - root node)
```

## Analysis

### What Rust Got Right ✓
- Extremely fast DAG generation (160ms vs 1.6s Julia vs ~1s Python)
- Extremely fast inference (232ms vs 69s Julia vs 472s Python)
- Total execution time: **0.88 seconds**

### What Rust Got Wrong ❌
- **Incorrect Bayesian scoring**: Different MAP structure and posterior probabilities
- Likely issues:
  - Categorical variable scoring simplified too much
  - Linear regression implementation may have bugs
  - Normalization or prior computation error

## Conclusion

**Julia is the validated winner:**
- ✅ Mathematically correct (matches Python exactly)
- ✅ 6.1x faster than Python
- ✅ Ready for Paper 1 experiments

**Rust potential (if bugs are fixed):**
- Could be **537x faster** than Python
- Could be **89x faster** than Julia
- But needs debugging to match Python/Julia results exactly

**Recommendation for Paper 1:**
Use Julia. It's validated, fast enough, and ready to use. Rust could be a future optimization if we need even more speed for very large datasets.
