# Experiment 001: Baseline Scalability

## Question
What's the baseline performance of Scientist AI beam search (Julia and Rust implementations) on medium-scale real-world data?

## Hypothesis
Both implementations should handle 50-100 variables within minutes and under 48GB RAM.

## Method
1. Create three SECOM dataset subsets: 50, 75, and 100 variables (from 590 total features)
2. Run Julia beam search implementation on each subset
3. Run Rust beam search implementation on each subset
4. Measure execution time and peak memory usage for each run

## Algorithm
- **Search strategy**: Beam search (beam_size=10, max_rounds=5)
- **Scoring**: Bayesian score with linear Gaussian models
- **Complexity penalty**: λ = 2.0

Note: Exhaustive search is intractable for 50+ variables (number of DAGs grows super-exponentially).

## Dataset
- Source: SECOM (Semiconductor Manufacturing)
- Full dataset: 590 features, 1,567 examples
- Subsets: 50, 75, 100 randomly selected features (seed=42 for reproducibility)

## Success Criteria
- Runtime: Minutes (not hours)
- Memory: <48GB peak RAM
- Both implementations complete successfully

## Results
[To be filled after experiments run]
