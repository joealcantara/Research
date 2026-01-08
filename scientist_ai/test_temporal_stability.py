#!/usr/bin/env python3
"""
Test temporal stability theory: Are features more stable in local time windows?

Compare global variance (all 574 eras) vs. local variance (rolling windows).
"""

import pandas as pd
import numpy as np

# Configuration
RANKING_FILE = "experiments/numerai_feature_ranks_all_eras.csv"
WINDOW_SIZE = 10  # Size of rolling window

def main():
    print("="*80)
    print("TEMPORAL STABILITY ANALYSIS")
    print("="*80)

    # Load ranking table
    print(f"\nLoading {RANKING_FILE}...")
    df = pd.read_csv(RANKING_FILE)
    rank_cols = [col for col in df.columns if col.startswith('rank_')]
    n_eras = len(rank_cols)
    print(f"Loaded {len(df)} features across {n_eras} eras")

    # Calculate global statistics (across all eras)
    print("\nCalculating global statistics...")
    df['global_mean'] = df[rank_cols].mean(axis=1)
    df['global_std'] = df[rank_cols].std(axis=1)

    print(f"  Global mean rank: {df['global_mean'].mean():.1f}")
    print(f"  Global std (avg across features): {df['global_std'].mean():.1f}")

    # Calculate local statistics (rolling window)
    print(f"\nCalculating local statistics (window size = {WINDOW_SIZE} eras)...")

    # For each feature, calculate std in each rolling window
    local_stds = []

    for idx, row in df.iterrows():
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(df)} features...")

        ranks = row[rank_cols].values

        # Calculate std for each rolling window
        window_stds = []
        for i in range(len(ranks) - WINDOW_SIZE + 1):
            window = ranks[i:i + WINDOW_SIZE]
            window_stds.append(np.std(window))

        # Average std across all windows for this feature
        avg_local_std = np.mean(window_stds)
        local_stds.append(avg_local_std)

    df['local_std'] = local_stds

    print(f"  Local std (avg across features): {df['local_std'].mean():.1f}")

    # Analysis
    print("\n" + "="*80)
    print("COMPARISON: GLOBAL vs LOCAL VARIANCE")
    print("="*80)

    global_avg_std = df['global_std'].mean()
    local_avg_std = df['local_std'].mean()

    print(f"\nAverage global std: {global_avg_std:.1f}")
    print(f"Average local std ({WINDOW_SIZE}-era window): {local_avg_std:.1f}")
    print(f"Ratio (local/global): {local_avg_std/global_avg_std:.3f}")

    if local_avg_std < global_avg_std:
        reduction = (1 - local_avg_std/global_avg_std) * 100
        print(f"\n✓ Theory confirmed: Local variance is {reduction:.1f}% lower than global variance")
        print(f"  Features ARE more stable in short time windows!")
    else:
        increase = (local_avg_std/global_avg_std - 1) * 100
        print(f"\n✗ Theory rejected: Local variance is {increase:.1f}% higher than global variance")
        print(f"  Features are NOT more stable in short time windows")

    # Show examples
    print("\n" + "="*80)
    print("EXAMPLES: Features with biggest stability difference")
    print("="*80)

    df['stability_gain'] = df['global_std'] - df['local_std']
    df['stability_ratio'] = df['local_std'] / df['global_std']

    # Features that are MUCH more stable locally
    print("\nTop 10 features with best local stability (lowest local/global ratio):")
    best_local = df.nsmallest(10, 'stability_ratio')
    for idx, row in best_local.iterrows():
        print(f"  {row['feature'][:50]:50s} - global std: {row['global_std']:6.1f}, local std: {row['local_std']:6.1f}, ratio: {row['stability_ratio']:.3f}")

    # Features that are actually LESS stable locally (worse in short windows)
    print("\nTop 10 features with worst local stability (highest local/global ratio):")
    worst_local = df.nlargest(10, 'stability_ratio')
    for idx, row in worst_local.iterrows():
        print(f"  {row['feature'][:50]:50s} - global std: {row['global_std']:6.1f}, local std: {row['local_std']:6.1f}, ratio: {row['stability_ratio']:.3f}")

    # Distribution analysis
    print("\n" + "="*80)
    print("DISTRIBUTION ANALYSIS")
    print("="*80)

    print(f"\nPercentage of features more stable locally (ratio < 1.0): {(df['stability_ratio'] < 1.0).sum() / len(df) * 100:.1f}%")
    print(f"Percentage of features less stable locally (ratio > 1.0): {(df['stability_ratio'] > 1.0).sum() / len(df) * 100:.1f}%")

    # Save results
    output_file = "experiments/temporal_stability_analysis.csv"
    df[['feature', 'global_mean', 'global_std', 'local_std', 'stability_ratio', 'stability_gain']].to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
