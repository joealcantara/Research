#!/usr/bin/env python3
"""
Analyze a single era and output its feature rankings.
Usage: python analyze_single_era.py <era_id>
Example: python analyze_single_era.py 0050
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sys

# Configuration
PARQUET_FILE = "/Users/joe/Documents/research/datasets/numerai/r1174__v5_2_train.parquet"

def analyze_era_features(df, era_id):
    """Run RandomForest on a single era and return feature rankings."""
    print(f"\n{'='*80}")
    print(f"ANALYZING ERA {era_id}")
    print(f"{'='*80}")

    # Filter to this era
    era_df = df[df['era'] == era_id].copy()
    n_obs = len(era_df)
    print(f"\nObservations: {n_obs:,}")

    if n_obs < 1000:
        print(f"ERROR: Only {n_obs} observations - insufficient data")
        sys.exit(1)

    # Get feature columns (exclude 'era' and all target columns)
    feature_cols = [col for col in era_df.columns
                    if col.startswith('feature_')]
    target_col = 'target'

    print(f"Features: {len(feature_cols):,}")

    # Prepare data
    X = era_df[feature_cols].values
    y = era_df[target_col].values

    # Remove any NaN rows
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    print(f"Valid observations: {len(X):,}")

    # Train RandomForest (NO max_depth limit - let trees grow fully)
    print(f"\nTraining RandomForest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    # Get R² score
    score = rf.score(X, y)
    print(f"R² = {score:.4f}")

    # Get feature importances and rank them
    importances = rf.feature_importances_

    # Create ranking (1 = most important)
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    })
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    feature_importance_df['rank'] = range(1, len(feature_importance_df) + 1)

    print(f"\nTop 5 features:")
    for i in range(5):
        print(f"  {i+1}. {feature_importance_df.iloc[i]['feature']}")
        print(f"     importance={feature_importance_df.iloc[i]['importance']:.6f}")

    return feature_importance_df[['feature', 'rank']]


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_single_era.py <era_id>")
        print("Example: python analyze_single_era.py 0050")
        sys.exit(1)

    era_id = sys.argv[1]

    # Load dataset
    print(f"\nLoading dataset from {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Loaded {len(df):,} observations")

    # Check if era exists
    available_eras = sorted(df['era'].unique())
    if era_id not in available_eras:
        print(f"\nERROR: Era {era_id} not found in dataset")
        print(f"Available eras: {available_eras[0]} to {available_eras[-1]}")
        sys.exit(1)

    # Analyze the era
    ranking_df = analyze_era_features(df, era_id)

    # Save to CSV
    output_file = f"experiments/numerai_feature_ranks_era{era_id}.csv"
    ranking_df.columns = ['feature', f'rank_{era_id}']
    ranking_df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}")
    print(f"\nSaved to: {output_file}")
    print(f"Features: {len(ranking_df)}")


if __name__ == "__main__":
    main()
