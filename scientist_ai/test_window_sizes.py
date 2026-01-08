#!/usr/bin/env python3
"""
Test different window sizes to find optimal temporal stability window.

Compare variance reduction across multiple window sizes.
"""

import pandas as pd
import numpy as np

# Configuration
RANKING_FILE = "experiments/numerai_feature_ranks_all_eras.csv"
WINDOW_SIZES = [5, 10, 20, 30, 50, 75, 100, 150, 200]  # Different window sizes to test

def calculate_local_std(ranks, window_size):
    """Calculate average local std across all rolling windows for a feature."""
    window_stds = []
    for i in range(len(ranks) - window_size + 1):
        window = ranks[i:i + window_size]
        window_stds.append(np.std(window))
    return np.mean(window_stds)


def main():
    print("="*80)
    print("WINDOW SIZE OPTIMIZATION ANALYSIS")
    print("="*80)

    # Load ranking table
    print(f"\nLoading {RANKING_FILE}...")
    df = pd.read_csv(RANKING_FILE)
    rank_cols = [col for col in df.columns if col.startswith('rank_')]
    n_eras = len(rank_cols)
    print(f"Loaded {len(df)} features across {n_eras} eras")

    # Calculate global statistics once
    print("\nCalculating global statistics...")
    df['global_mean'] = df[rank_cols].mean(axis=1)
    df['global_std'] = df[rank_cols].std(axis=1)
    global_avg_std = df['global_std'].mean()

    print(f"  Global mean rank: {df['global_mean'].mean():.1f}")
    print(f"  Global std (avg across features): {global_avg_std:.1f}")

    # Test each window size
    results = []

    for window_size in WINDOW_SIZES:
        print(f"\nTesting window size = {window_size} eras...")

        # Calculate local std for all features
        local_stds = []
        for idx, row in df.iterrows():
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(df)} features...")

            ranks = row[rank_cols].values
            avg_local_std = calculate_local_std(ranks, window_size)
            local_stds.append(avg_local_std)

        local_avg_std = np.mean(local_stds)
        ratio = local_avg_std / global_avg_std
        reduction = (1 - ratio) * 100

        results.append({
            'window_size': window_size,
            'local_avg_std': local_avg_std,
            'local_global_ratio': ratio,
            'variance_reduction_pct': reduction
        })

        print(f"  Local avg std: {local_avg_std:.1f}")
        print(f"  Ratio (local/global): {ratio:.3f}")
        print(f"  Variance reduction: {reduction:.1f}%")

    # Analysis
    print("\n" + "="*80)
    print("SUMMARY: WINDOW SIZE vs VARIANCE REDUCTION")
    print("="*80)

    results_df = pd.DataFrame(results)
    print(f"\n{'Window Size':>12} | {'Local Avg Std':>14} | {'L/G Ratio':>10} | {'Variance Reduction':>18}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['window_size']:>12} | {row['local_avg_std']:>14.1f} | {row['local_global_ratio']:>10.3f} | {row['variance_reduction_pct']:>17.1f}%")

    # Find optimal window
    best_idx = results_df['variance_reduction_pct'].idxmax()
    best_window = results_df.loc[best_idx]

    print("\n" + "="*80)
    print("OPTIMAL WINDOW SIZE")
    print("="*80)

    print(f"\nBest window size: {int(best_window['window_size'])} eras")
    print(f"  Variance reduction: {best_window['variance_reduction_pct']:.1f}%")
    print(f"  Local avg std: {best_window['local_avg_std']:.1f}")
    print(f"  Local/global ratio: {best_window['local_global_ratio']:.3f}")

    # Visualize the trend
    print("\n" + "="*80)
    print("VARIANCE REDUCTION vs WINDOW SIZE")
    print("="*80)
    print("\n" + " " * 20 + "Variance Reduction (%)")
    print(" " * 10 + "0      10     20     30     40     50     60")
    print(" " * 10 + "|------|------|------|------|------|------|")

    max_reduction = results_df['variance_reduction_pct'].max()
    for _, row in results_df.iterrows():
        pct = row['variance_reduction_pct']
        bar_length = int((pct / 60) * 50)  # Scale to 50 chars max
        bar = "█" * bar_length
        print(f"{int(row['window_size']):>4} eras: |{bar:<50}| {pct:5.1f}%")

    # Save results
    output_file = "experiments/window_size_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
