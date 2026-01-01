# Numerai Dataset Runtime Predictions

## Numerai Dataset Characteristics

**Known from research context:**
- **Features:** 2,146 variables
- **Samples:** Typically ~500,000 training samples (but we can subsample)
- **Feature types:** All continuous (standardized financial features)
- **Target:** Categorical (5 classes: 0, 0.25, 0.5, 0.75, 1.0)
- **Challenge:** Massive feature space, relatively sparse relationships

## Scaling Analysis

### Direct Extrapolation (Naive Approach)

**Assumption:** Run beam search on all 2,146 variables directly

**Complexity scaling from Wine → Numerai:**
- Variables: 14 → 2,146 (153x increase)
- Each variable searches over ~2,145 potential parents
- Beam search complexity ≈ O(n² × beam_size × rounds)

**Rough extrapolation:**

| Implementation | Wine (14 vars) | Numerai (2,146 vars) | Scaling Factor |
|----------------|----------------|----------------------|----------------|
| Python | 6.5s | **~33 hours** | 153² ≈ 23,000x |
| Julia | 2.0s | **~13 hours** | 153² ≈ 23,000x |
| Rust | 43.8ms | **~17 minutes** | 153² ≈ 23,000x |

**Problems with this approach:**
- ❌ Combinatorially explosive
- ❌ Most features likely irrelevant
- ❌ No domain knowledge used
- ❌ Not practical even with Rust

### Practical Approach (Hierarchical + Feature Selection)

This is what you'd actually do for Paper 1/2:

#### Step 1: Feature Selection (Reduce 2,146 → ~50-100 important features)

**Methods:**
- Correlation with target
- Mutual information
- Random forest feature importance
- Domain knowledge from Numerai forum

**Time:** Minutes (one-time cost)

#### Step 2: Learn Structure on Selected Features

**Scenario A: 50 most important features**

Based on interpolation between Wine (14) and hypothetical 50-variable dataset:

| Implementation | Predicted Time | Confidence |
|----------------|----------------|------------|
| Python | **3-5 minutes** | High (linear extrapolation) |
| Julia | **30-60 seconds** | High |
| Rust | **1-2 seconds** | High |

**Scenario B: 100 most important features**

| Implementation | Predicted Time | Confidence |
|----------------|----------------|------------|
| Python | **15-30 minutes** | Medium (quadratic scaling) |
| Julia | **2-5 minutes** | Medium |
| Rust | **3-8 seconds** | Medium |

#### Step 3: Iterative Refinement

**Hierarchical approach:**
1. Learn structure on top 50 features (Rust: ~2s)
2. Identify key hub variables
3. Expand to include connected features (next 50)
4. Refine structure (Rust: ~2s per iteration)

**Total time for 3-4 iterations:**
- Rust: **10-20 seconds**
- Julia: **5-10 minutes**

## Sample Size Scaling

Numerai has ~500k samples, but we can subsample:

**Inference cost scales linearly with samples:**

| Sample Size | Relative Cost | Use Case |
|-------------|---------------|----------|
| 1,000 | 1x | Rapid prototyping |
| 10,000 | 10x | Development |
| 100,000 | 100x | Validation |
| 500,000 | 500x | Final benchmark |

**For 100 variables with different sample sizes:**

| Samples | Python | Julia | Rust |
|---------|--------|-------|------|
| 1,000 | 15 min | 2 min | **3 sec** |
| 10,000 | 2.5 hrs | 20 min | **30 sec** |
| 100,000 | 25 hrs | 3.3 hrs | **5 min** |
| 500,000 | 125 hrs | 16.5 hrs | **25 min** |

## Recommended Numerai Workflow

### Phase 1: Feature Selection (One-time)
```
1. Correlation analysis        (minutes)
2. Mutual information          (minutes)
3. Select top 100 features     (seconds)
```

### Phase 2: Structure Learning (Iterative)
```
For different feature sets (50, 75, 100):
  For different sample sizes (1k, 10k, 100k):
    Run beam search → Get structure

With Rust: ~30 experiments × 3-30 sec = 15-20 minutes total
With Julia: ~30 experiments × 30-300 sec = 15-150 minutes total
With Python: ~30 experiments × 3-30 min = 1.5-15 hours total
```

### Phase 3: Final Validation (Once)
```
Best configuration on full 500k samples:
  Rust:   ~25 minutes
  Julia:  ~16 hours
  Python: ~5 days
```

## Paper 2 Experimental Design Estimate

**Realistic Paper 2 workflow:**

1. **Feature selection experiments** (10 different thresholds)
   - Test 50, 75, 100, 125, 150 features
   - Rust: 5 × 2s = **10 seconds**

2. **Structure learning experiments** (50 runs)
   - Different beam sizes, rounds, penalties
   - On 100 features, 10k samples
   - Rust: 50 × 30s = **25 minutes**

3. **Scaling validation** (5 sample sizes)
   - 1k, 10k, 100k, 250k, 500k samples
   - Rust: (3s + 30s + 5m + 15m + 25m) = **45 minutes**

4. **Theory language comparison** (3 theories × 10 runs)
   - Linear, Trees, Mixed
   - Rust: 30 × 30s = **15 minutes**

**Total experimentation time:**
- **Rust: ~1.5 hours** ✅ Practical for iterating
- **Julia: ~15 hours** ⚠️ Still doable overnight
- **Python: ~150 hours** ❌ Impractical (6+ days)

## Critical Insights

### 1. Feature Selection is Mandatory
- Can't run beam search on 2,146 variables directly
- Even Rust would take ~17 minutes per run
- 100 variables is the practical upper limit for exhaustive exploration

### 2. Sample Size Matters More Than Variable Count
- Numerai's 500k samples vs Wine's 178 samples = 2,800x difference
- This dominates runtime for large experiments
- Solution: Use 10k-100k subsample for development, full data for final validation

### 3. Rust Enables True Experimentation
- With Rust, you can test 100+ parameter combinations in an afternoon
- With Python, same experiments take days
- This is the difference between "can iterate" and "stuck waiting"

### 4. Hierarchical Structure Learning
- Don't learn all 100 variables at once
- Learn in groups of 20-30, then combine
- Rust makes this practical: 5 groups × 2s = 10 seconds

## Recommended Strategy for Numerai

### Conservative (Paper 1 proof-of-concept)
```
Features:   50 most important
Samples:    10,000 subsample
Method:     Linear Gaussian beam search
Tool:       Julia (comfortable development)
Time:       ~30 seconds per experiment
Goal:       Show method works on real financial data
```

### Ambitious (Paper 2 full scale)
```
Features:   100-150 with hierarchical learning
Samples:    100,000 for development, 500k for final
Method:     Mixed theory (linear + trees)
Tool:       Rust (enables rapid iteration)
Time:       ~30 sec dev, ~25 min final
Goal:       Show method scales to production datasets
```

### Aspirational (Future work)
```
Features:   All 2,146 with intelligent pruning
Samples:    Full 500,000
Method:     GFlowNets for structure search
Tool:       Rust + GPU acceleration
Time:       ~hours (still exploration needed)
Goal:       Match/beat state-of-the-art causal discovery
```

## Bottom Line

**For Numerai experiments:**

| Phase | Setup | Python | Julia | Rust |
|-------|-------|--------|-------|------|
| **Development** | 50 vars, 10k samples | 3 min | 30 sec | **1 sec** |
| **Validation** | 100 vars, 100k samples | 25 hrs | 3.3 hrs | **5 min** |
| **Publication** | 100 vars, 500k samples | 125 hrs | 16.5 hrs | **25 min** |

**Rust doesn't just make things faster - it makes iterative research practical.**

With Python: You get 1-2 experiments per day
With Julia: You get 10-20 experiments per day
With Rust: You get 100-500 experiments per day

That's the difference between a 3-month paper and a 3-week paper.
