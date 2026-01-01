# Wine Dataset Performance Predictions

## Dataset Comparison

| Dataset | Variables | Samples | DAG Structures |
|---------|-----------|---------|----------------|
| Iris | 5 | 150 | 29,281 (exhaustive) |
| Wine | 14 | 178 | ~1.5 trillion (theoretical) |

## Actual Wine Beam Search Results (Linear Gaussian)

We already ran Wine with beam search:

| Implementation | Actual Time | Method |
|----------------|-------------|--------|
| Python | 6.5s | Beam search (size=10, rounds=5) |
| Julia | 2.0s | Beam search (size=10, rounds=5) |
| Rust | **predicted** | Not run yet |

### Rust Wine Beam Search Prediction

Based on Iris speedup ratios:
- Rust is **498x faster** than Python on Iris
- Rust is **82x faster** than Julia on Iris

**Conservative estimate (accounting for beam search overhead):**
- If we assume Rust maintains even **50%** of its Iris speedup on Wine:
  - vs Python: 6.5s / 249 = **0.026s (26ms)**
  - vs Julia: 2.0s / 41 = **0.049s (49ms)**

**More realistic estimate (70% of Iris speedup):**
- vs Python: 6.5s / 348 = **0.019s (19ms)**
- vs Julia: 2.0s / 57 = **0.035s (35ms)**

**Predicted Rust Wine (beam search): ~20-50ms**

## Theoretical Wine Exhaustive Enumeration

### Number of DAG Structures by Variable Count

The number of DAGs grows super-exponentially:

| Variables | Approx DAG Count | Source |
|-----------|------------------|--------|
| 3 | 25 | Exact |
| 4 | 543 | Exact |
| 5 | 29,281 | Exact (our benchmark) |
| 6 | 3,781,503 | Calculated |
| 7 | ~1.4 billion | Estimated |
| 10 | ~1 trillion | Estimated |
| 14 | ~1.5 quadrillion | Estimated |

**Wine exhaustive enumeration is impractical** - we'd need to score ~1.5 quadrillion structures.

### Time Estimates (if we could do it)

Based on Iris inference speed per structure:

**Inference time per structure:**
- Python: 472s / 29,281 = 16.1 ms/structure
- Julia: 69s / 29,281 = 2.4 ms/structure
- Rust: 0.30s / 29,281 = 0.010 ms/structure

**For 1.5 quadrillion Wine structures:**

| Implementation | Time per Structure | Total Time |
|----------------|-------------------|------------|
| Python | 16.1 ms | **764,000 years** |
| Julia | 2.4 ms | **114,000 years** |
| Rust | 0.010 ms | **475 years** |

Even Rust can't save us from exponential explosion! 😅

## Practical Predictions for Paper 1

### Medium-scale datasets (20-50 variables, beam search)

**Assumptions:**
- Beam search: size=20, rounds=10
- ~200-500 samples
- Linear Gaussian models

| Implementation | Estimated Time |
|----------------|----------------|
| Python | 30s - 5min |
| Julia | 5s - 30s |
| Rust | **0.1s - 1s** |

### Large-scale datasets (50-100 variables, beam search)

**Assumptions:**
- Beam search: size=20, rounds=10
- ~500-1000 samples
- Mixed theory (linear + trees)

| Implementation | Estimated Time |
|----------------|----------------|
| Python | 5min - 1hr |
| Julia | 30s - 5min |
| Rust | **1s - 10s** |

### Numerai (2,146 features) - Paper 2 territory

**Assumptions:**
- Aggressive pruning/beam search
- Feature selection first (reduce to ~100-200 most important)
- Iterative refinement

| Implementation | Estimated Time |
|----------------|----------------|
| Python | Hours - Days |
| Julia | Minutes - Hours |
| Rust | **Seconds - Minutes** |

## Recommendation

**For Wine specifically:**
Let's **run the Rust Wine benchmark** to validate our predictions!

Expected result: **~20-50ms** (compared to Julia's 2s, Python's 6.5s)

This would give us:
- Python → Rust: **130-325x speedup**
- Julia → Rust: **40-100x speedup**

## Key Insights

1. **Beam search scales well** - Linear in beam size, not exponential
2. **Rust shines on inference** - 230x faster than Julia on Iris inference
3. **For Paper 1 (50-100 vars)**: Julia is fast enough, but Rust gives huge margin
4. **For Paper 2 (1000+ vars)**: Rust becomes critical for iterating quickly

Want to run Rust on Wine to validate these predictions?
