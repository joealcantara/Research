#!/usr/bin/env python3
"""
Compare learned structures against ground truth for synthetic datasets.

Calculates structural accuracy metrics:
- True Positives (correctly identified edges)
- False Positives (incorrectly added edges)
- False Negatives (missed edges)
- Precision, Recall, F1 score
- Structural Hamming Distance (SHD)
"""

import re
import json
from pathlib import Path
from typing import Dict, Set, Tuple


def load_true_structure(metadata_file: Path) -> Dict[int, Set[int]]:
    """Load true DAG structure from metadata file."""
    structure = {}

    with open(metadata_file, 'r') as f:
        lines = f.readlines()

    # Find the "True Structure:" section
    in_structure = False
    for line in lines:
        if "True Structure:" in line:
            in_structure = True
            continue

        if in_structure and line.strip():
            # Parse lines like "  0: (no parents)" or "  1: ← [0]"
            match = re.match(r'\s+(\d+):\s*(?:←\s*\[([^\]]*)\]|\(no parents\))', line)
            if match:
                node = int(match.group(1))
                parents_str = match.group(2)

                if parents_str:
                    # Parse parent list
                    parents = {int(p.strip()) for p in parents_str.split(',')}
                else:
                    parents = set()

                structure[node] = parents

    return structure


def load_learned_structure(result_file: Path) -> Dict[int, Set[int]]:
    """Load learned DAG structure from benchmark result file."""
    structure = {}

    with open(result_file, 'r') as f:
        lines = f.readlines()

    # Find structure section
    in_structure = False
    for line in lines:
        if "Best structure:" in line:
            in_structure = True
            continue

        if in_structure and line.strip() and not line.startswith('Results saved'):
            # Parse lines like "  Column42 ← [:Column41, :Column310, ...]"
            # or "  var_0: ← [var_5, var_10]"
            match = re.match(r'\s+(?:Column|var_)(\d+)\s*[←:]+\s*\[([^\]]*)\]', line)
            if match:
                node = int(match.group(1))
                parents_str = match.group(2)

                # Parse parent list
                parents = set()
                if parents_str.strip():
                    for p in parents_str.split(','):
                        p = p.strip().strip(':')
                        # Extract number from :Column123 or :var_5
                        parent_match = re.search(r'(\d+)', p)
                        if parent_match:
                            parents.add(int(parent_match.group(1)))

                structure[node] = parents

    return structure


def compare_structures(true: Dict[int, Set[int]], learned: Dict[int, Set[int]]) -> Dict:
    """Compare two structures and calculate metrics."""

    # Ensure both have same set of nodes
    all_nodes = set(true.keys()) | set(learned.keys())

    # Count edges
    true_edges = {(node, parent) for node, parents in true.items() for parent in parents}
    learned_edges = {(node, parent) for node, parents in learned.items() for parent in parents}

    tp = len(true_edges & learned_edges)  # True positives
    fp = len(learned_edges - true_edges)   # False positives
    fn = len(true_edges - learned_edges)   # False negatives

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Structural Hamming Distance
    shd = fp + fn

    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'shd': shd,
        'true_edges': len(true_edges),
        'learned_edges': len(learned_edges),
    }


def main():
    """Compare all synthetic datasets."""

    datasets_dir = Path("synthetic_datasets")
    results_dir = Path("results")

    # Find all synthetic datasets
    dataset_names = []
    for csv_file in sorted(datasets_dir.glob("*.csv")):
        dataset_names.append(csv_file.stem)

    print("\n" + "="*70)
    print("Ground Truth Comparison: Synthetic Datasets")
    print("="*70 + "\n")

    total_tp = total_fp = total_fn = 0
    perfect_recoveries = 0
    results_found = 0

    for dataset in dataset_names:
        metadata_file = datasets_dir / f"{dataset}_metadata.txt"

        if not metadata_file.exists():
            print(f"⊘ {dataset}: no metadata found")
            continue

        # Load true structure
        true_structure = load_true_structure(metadata_file)

        # Check both Rust and Julia results
        for impl in ['rust', 'julia']:
            result_file = results_dir / f"{impl}_{dataset}.txt"

            if not result_file.exists():
                continue

            results_found += 1
            learned_structure = load_learned_structure(result_file)
            metrics = compare_structures(true_structure, learned_structure)

            # Print results
            tp, fp, fn = metrics['true_positives'], metrics['false_positives'], metrics['false_negatives']
            precision, recall, f1 = metrics['precision'], metrics['recall'], metrics['f1_score']
            shd = metrics['shd']

            total_tp += tp
            total_fp += fp
            total_fn += fn

            status = "✓" if shd == 0 else "✗"
            if shd == 0:
                perfect_recoveries += 1

            impl_label = impl.capitalize().ljust(6)
            print(f"{status} {dataset.ljust(25)} ({impl_label}): "
                  f"TP={tp:2d} FP={fp:2d} FN={fn:2d} | "
                  f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} | SHD={shd:2d}")

    # Summary
    print("\n" + "-"*70)
    print(f"Summary: {results_found} results analyzed")
    print(f"  Perfect recoveries (SHD=0): {perfect_recoveries}/{results_found}")
    print(f"  Total edges: TP={total_tp}, FP={total_fp}, FN={total_fn}")

    if total_tp + total_fp > 0:
        overall_precision = total_tp / (total_tp + total_fp)
        print(f"  Overall precision: {overall_precision:.3f}")

    if total_tp + total_fn > 0:
        overall_recall = total_tp / (total_tp + total_fn)
        print(f"  Overall recall: {overall_recall:.3f}")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
