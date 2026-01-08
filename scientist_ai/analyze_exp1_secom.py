#!/usr/bin/env python3
"""Analyze Experiment 1 SECOM results: beam search vs exhaustive ground truth"""

import json

# Optimal score from exhaustive search (4 structures tied)
OPTIMAL_SCORE = -15405.555

# Load all run results
results = []
for i in range(1, 21):
    path = f"results/exp1_secom/run_{i}.json"
    try:
        with open(path) as f:
            data = json.load(f)
            score = data['methods'][0]['score']
            structure = data['methods'][0]['best_structure']
            edges = data['methods'][0]['edges']
            results.append({
                'run': i,
                'score': score,
                'edges': edges,
                'structure': structure
            })
    except FileNotFoundError:
        print(f"Warning: {path} not found")

# Analyze
optimal_count = sum(1 for r in results if abs(r['score'] - OPTIMAL_SCORE) < 0.01)
scores = [r['score'] for r in results]
gaps = [r['score'] - OPTIMAL_SCORE for r in results]

print("=" * 60)
print("EXPERIMENT 1: GROUND TRUTH VALIDATION (SECOM)")
print("=" * 60)
print(f"Dataset: secom_4var (4 continuous variables, 1567 samples)")
print(f"Exhaustive optimal: {OPTIMAL_SCORE:.3f} (4 structures tied)")
print(f"Beam search runs: {len(results)}")
print()

print("=" * 60)
print("SUCCESS RATE")
print("=" * 60)
print(f"Found optimal (within 0.01): {optimal_count}/{len(results)} ({100*optimal_count/len(results):.1f}%)")
print()

print("=" * 60)
print("SCORE STATISTICS")
print("=" * 60)
print(f"Best score found:  {min(scores):.6f}")
print(f"Worst score found: {max(scores):.6f}")
print(f"Mean score:        {sum(scores)/len(scores):.6f}")
print(f"Mean gap from opt: {sum(gaps)/len(gaps):.6f}")
print(f"Max gap from opt:  {max(gaps):.6f}")
print()

print("=" * 60)
print("ALL RUNS")
print("=" * 60)
for r in results:
    gap = r['score'] - OPTIMAL_SCORE
    optimal_mark = " ✓ OPTIMAL" if abs(gap) < 0.01 else f" (gap: {gap:+.3f})"
    print(f"Run {r['run']:2d}: score={r['score']:.6f}, edges={r['edges']}{optimal_mark}")
print()

print("=" * 60)
print("CONCLUSION")
print("=" * 60)
if optimal_count == len(results):
    print(f"✓ Beam search found optimal structure on ALL {len(results)} runs")
elif optimal_count >= len(results) * 0.9:
    print(f"✓ Beam search found optimal on {optimal_count}/{len(results)} runs ({100*optimal_count/len(results):.1f}%)")
    print(f"  High success rate demonstrates algorithm quality")
else:
    print(f"⚠ Beam search found optimal on {optimal_count}/{len(results)} runs ({100*optimal_count/len(results):.1f}%)")
    print(f"  Success rate may need improvement")
