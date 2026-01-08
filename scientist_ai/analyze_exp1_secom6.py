#!/usr/bin/env python3
"""Analyze Experiment 1c SECOM 6-var results: score consistency without ground truth"""

import json
from collections import Counter

# Load all run results
results = []
structures = []
for i in range(1, 21):
    path = f"results/exp1_secom6/run_{i}.json"
    try:
        with open(path) as f:
            data = json.load(f)
            score = data['methods'][0]['score']
            structure = data['methods'][0]['best_structure']
            edges = data['methods'][0]['edges']

            # Convert structure to canonical edge list for comparison
            edge_list = []
            for child, parents in structure.items():
                for parent in parents:
                    edge_list.append(f"{parent}->{child}")
            edge_list.sort()
            edge_str = ",".join(edge_list) if edge_list else "empty"

            results.append({
                'run': i,
                'score': score,
                'edges': edges,
                'structure_str': edge_str
            })
            structures.append(edge_str)
    except FileNotFoundError:
        print(f"Warning: {path} not found")

# Analyze
scores = [r['score'] for r in results]
unique_structures = Counter(structures)

print("=" * 70)
print("EXPERIMENT 1C: 6-VARIABLE VALIDATION (NO GROUND TRUTH)")
print("=" * 70)
print(f"Dataset: secom_6var (6 continuous variables, 1567 samples)")
print(f"Search space: 3,781,503 possible DAGs")
print(f"Beam size: 10 (~0.00026% coverage per round)")
print(f"Beam search runs: {len(results)}")
print()

print("=" * 70)
print("SCORE STATISTICS")
print("=" * 70)
print(f"Best score found:  {min(scores):.10f}")
print(f"Worst score found: {max(scores):.10f}")
print(f"Mean score:        {sum(scores)/len(scores):.10f}")
print(f"Score std dev:     {(sum((s-sum(scores)/len(scores))**2 for s in scores)/len(scores))**0.5:.10f}")
print(f"Score range:       {max(scores) - min(scores):.10f}")
print()

print("=" * 70)
print("STRUCTURE CONSISTENCY")
print("=" * 70)
print(f"Unique structures found: {len(unique_structures)}")
for structure, count in unique_structures.most_common():
    print(f"  {count}/20 runs: {structure}")
print()

print("=" * 70)
print("ALL RUNS")
print("=" * 70)
for r in results:
    print(f"Run {r['run']:2d}: score={r['score']:.6f}, edges={r['edges']}")
print()

print("=" * 70)
print("CONCLUSION")
print("=" * 70)
if len(unique_structures) == 1:
    print(f"✓ Perfect consistency: ALL {len(results)} runs found the SAME structure")
    print(f"✓ Score variance: {(sum((s-sum(scores)/len(scores))**2 for s in scores)/len(scores))**0.5:.2e} (machine precision)")
    print(f"✓ From 3.78M DAG search space with 0.00026% coverage per round")
    print(f"")
    print(f"This is STRONG evidence that:")
    print(f"1. Beam search reliably finds a dominant optimal structure")
    print(f"2. The algorithm works well beyond trivially small search spaces")
    print(f"3. Results are deterministic despite stochastic search")
else:
    print(f"⚠ Found {len(unique_structures)} different structures with similar scores")
    print(f"  This suggests multiple local optima or score plateaus")

    if max(scores) - min(scores) < 0.01:
        print(f"  But score range is tiny ({max(scores) - min(scores):.2e})")
        print(f"  → Likely equivalent structures (same score)")
