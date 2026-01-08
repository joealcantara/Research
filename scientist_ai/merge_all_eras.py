#!/usr/bin/env python3
"""
Merge all era rankings into a single table.
This is the master merge script - updates whenever new eras are added.
"""

import pandas as pd

# All eras (1-574)
ERAS = [f'{i:04d}' for i in range(1, 575)]
OUTPUT_RANKING_FILE = "experiments/numerai_feature_ranks_all_eras.csv"

def main():
    print("="*80)
    print(f"MERGING {len(ERAS)}-ERA FEATURE RANKING TABLE")
    print("="*80)

    # Load all individual era files
    print("\nLoading all era rankings...")

    all_dfs = []
    for i, era in enumerate(ERAS):
        era_file = f"experiments/numerai_feature_ranks_era{era}.csv"
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(ERAS)} eras...")
        era_df = pd.read_csv(era_file)
        all_dfs.append(era_df)

    print(f"  Loaded {len(ERAS)}/{len(ERAS)} eras")

    # Merge all dataframes
    print("\nMerging all eras...")
    combined_df = all_dfs[0].copy()
    for i, era_df in enumerate(all_dfs[1:], 1):
        if (i + 1) % 100 == 0:
            print(f"  Merged {i + 1}/{len(ERAS)} eras...")
        combined_df = combined_df.merge(
            era_df,
            on='feature',
            how='outer'
        )

    # Reorder columns to be chronological
    ordered_cols = ['feature'] + [f'rank_{era}' for era in ERAS]
    combined_df = combined_df[ordered_cols]

    # Save combined ranking table
    print(f"\nSaving {len(ERAS)}-era ranking table to {OUTPUT_RANKING_FILE}...")
    combined_df.to_csv(OUTPUT_RANKING_FILE, index=False)
    print(f"Saved {len(combined_df)} features with rankings for {len(ERAS)} eras")

    # Calculate statistics
    print("\n" + "="*80)
    print("FEATURE STABILITY ANALYSIS")
    print("="*80)

    rank_cols = [f'rank_{era}' for era in ERAS]

    # Calculate mean rank and std for each feature
    combined_df['mean_rank'] = combined_df[rank_cols].mean(axis=1)
    combined_df['std_rank'] = combined_df[rank_cols].std(axis=1)

    # Count how many eras each feature is in top 785
    combined_df['top785_count'] = (combined_df[rank_cols] <= 785).sum(axis=1)

    print(f"\nTotal eras analyzed: {len(ERAS)}")

    # Show distribution of top785 counts
    print("\nDistribution of features by # of eras in top 785:")
    for count in sorted(combined_df['top785_count'].unique(), reverse=True)[:10]:
        n_features = (combined_df['top785_count'] == count).sum()
        pct = 100 * n_features / len(combined_df)
        print(f"  {count:2d} eras: {n_features:4d} features ({pct:5.2f}%)")

    # Show features that are in top 785 for most eras
    print(f"\nTop 10 features by consistency (most eras in top 785):")
    top_consistent = combined_df.nlargest(10, 'top785_count')
    for idx, row in top_consistent.iterrows():
        print(f"  {row['feature'][:50]:50s} - {int(row['top785_count'])}/{len(ERAS)} eras (mean rank: {row['mean_rank']:.0f})")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nFile created:")
    print(f"  - {OUTPUT_RANKING_FILE} (ranking table for all {len(ERAS)} eras)")


if __name__ == "__main__":
    main()
