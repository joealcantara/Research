# Iris Benchmark Validation: Python vs Julia

## Exhaustive Enumeration (29,281 DAG structures)

### Performance Comparison

| Implementation | Time | Speedup |
|----------------|------|---------|
| Python | 7:53 (473 seconds) | 1.0x |
| Julia | 1:18 (78 seconds) | **6.1x faster** |

### Results Comparison

**Top 1 Posterior:**
- Python: 0.174838 (Structure 28708)
- Julia: 0.174838 (Structure 28709)
- Match: ✓ (structure indices offset by 1 due to 0-based vs 1-based indexing)

**Top 10 Cumulative:**
- Python: 0.763151 (76.32%)
- Julia: 0.763151 (76.32%)
- Match: ✓ Exact

**Top 100 Cumulative:**
- Python: 0.999990 (100.0%)
- Julia: 0.99999 (100.0%)
- Match: ✓ Exact

### MAP Structure Comparison

Both implementations found the **identical** MAP structure:

```
sepal_length: ← sepal_width, species
sepal_width: (no parents - root node)
petal_length: ← sepal_length, sepal_width, petal_width, species
petal_width: ← sepal_length, species
species: ← sepal_width
```

- Total edges: 9
- Posterior probability: 0.174838
- Log-posterior: -350.89

### Top 20 Structures Score Comparison

All top 20 structures match exactly between Python and Julia:

| Rank | Python Structure | Julia Structure | Posterior | Edges | Log-Post |
|------|------------------|-----------------|-----------|-------|----------|
| 1 | 28708 | 28709 | 0.1748 | 9 | -350.89 |
| 2 | 27696 | 27697 | 0.1147 | 8 | -351.31 |
| 3 | 29049 | 29050 | 0.1064 | 9 | -351.39 |
| 4 | 28877 | 28878 | 0.1064 | 9 | -351.39 |
| 5 | 26720 | 26721 | 0.0804 | 8 | -351.67 |
| 6 | 28765 | 28766 | 0.0747 | 9 | -351.74 |
| 7 | 27585 | 27586 | 0.0279 | 8 | -352.73 |
| 8 | 28976 | 28977 | 0.0259 | 9 | -352.80 |
| 9 | 28965 | 28966 | 0.0259 | 9 | -352.80 |
| 10 | 29017 | 29018 | 0.0259 | 9 | -352.80 |

Note: Structure indices differ by 1 (Python uses 0-based, Julia uses 1-based indexing)

### Edge Distribution

Both implementations generated identical edge distributions:

| Edges | Count |
|-------|-------|
| 0 | 1 |
| 1 | 20 |
| 2 | 180 |
| 3 | 940 |
| 4 | 3,050 |
| 5 | 6,180 |
| 6 | 7,960 |
| 7 | 6,540 |
| 8 | 3,330 |
| 9 | 960 |
| 10 | 120 |
| **Total** | **29,281** |

## Validation Status

✅ **PASSED** - Complete mathematical equivalence between Python and Julia implementations

- Identical MAP structure
- Identical posterior probabilities (all top 100 match exactly)
- Identical edge distributions
- 6.1x performance improvement

## Conclusion

The Julia port is **mathematically correct** and **significantly faster** than the Python implementation. All structure scores, posterior probabilities, and the MAP structure match exactly between implementations.

This validates the Julia port for Paper 1 scalability experiments.
