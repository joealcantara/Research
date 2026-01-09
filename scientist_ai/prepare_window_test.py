#!/usr/bin/env python3
"""
Extract one 5-era window from Numerai data with 596-feature subset.
This is a test run to benchmark causal discovery performance.
"""

import pandas as pd
import sys

# Configuration
PARQUET_FILE = "/Users/joe/Documents/research/datasets/numerai/r1174__v5_2_train.parquet"
FEATURE_LIST = "/Users/joe/Documents/research/code/scientist_ai/experiments/reduced_feature_set_596.txt"
OUTPUT_CSV = "/Users/joe/Documents/research/code/scientist_ai/experiments/window_test_5eras_596features.csv"

def load_feature_subset():
    """Load the 596-feature subset list."""
    features = []
    with open(FEATURE_LIST, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                features.append(line)
    return features

def main():
    print("\n" + "="*80)
    print("PREPARING 5-ERA WINDOW TEST")
    print("="*80)

    # Load feature subset
    print("\nLoading 596-feature subset...")
    feature_subset = load_feature_subset()
    print(f"Features to extract: {len(feature_subset)}")

    # Load Numerai data
    print(f"\nLoading Numerai training data from {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Total observations: {len(df):,}")

    # Get available eras
    available_eras = sorted(df['era'].unique())
    print(f"Available eras: {available_eras[0]} to {available_eras[-1]} ({len(available_eras)} eras)")

    # Select first 5 eras for the test
    test_eras = available_eras[:5]
    print(f"\nUsing eras: {test_eras}")

    # Filter to test eras
    window_df = df[df['era'].isin(test_eras)].copy()
    print(f"Window observations: {len(window_df):,}")

    # Verify all features exist in the dataset
    missing_features = [f for f in feature_subset if f not in window_df.columns]
    if missing_features:
        print(f"\nWARNING: {len(missing_features)} features not found in dataset")
        print(f"First 5 missing: {missing_features[:5]}")
        feature_subset = [f for f in feature_subset if f in window_df.columns]
        print(f"Continuing with {len(feature_subset)} available features")

    # Extract subset of features (no target, no era)
    subset_df = window_df[feature_subset].copy()

    # Check for NaN values
    nan_counts = subset_df.isna().sum()
    features_with_nan = nan_counts[nan_counts > 0]
    if len(features_with_nan) > 0:
        print(f"\nWARNING: {len(features_with_nan)} features have NaN values")
        print(f"Total NaN cells: {nan_counts.sum()}")
        # Drop rows with any NaN
        subset_df = subset_df.dropna()
        print(f"After dropping NaN rows: {len(subset_df):,} observations")

    # Export to CSV
    print(f"\nExporting to {OUTPUT_CSV}...")
    subset_df.to_csv(OUTPUT_CSV, index=False)

    # Report statistics
    file_size_mb = pd.read_csv(OUTPUT_CSV).memory_usage(deep=True).sum() / (1024**2)

    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"\nFile: {OUTPUT_CSV}")
    print(f"Observations: {len(subset_df):,}")
    print(f"Features: {len(subset_df.columns)}")
    print(f"Estimated CSV size: ~{file_size_mb:.1f} MB")
    print(f"\nReady for causal discovery with scientist_ai")

if __name__ == "__main__":
    main()
