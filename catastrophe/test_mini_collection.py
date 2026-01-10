"""Minimal test: Generate 3 tiers (honest + tier 1 + tier 6) for problem 0."""
import json
import sys
from dataset_utils import load_my_dataset
from tiered_coder import TieredCoder
from executor import run_test

print("Loading dataset...", flush=True)
dataset = load_my_dataset(split="train", difficulty="introductory")
sample = dataset[0]

print(f"\nProblem 0: {sample['problem_id']}", flush=True)

# Test just 3 tiers: honest, tier_1_obvious, tier_6_expert
test_tiers = ["honest", "tier_1_obvious", "tier_6_expert"]

results = {
    "problem_id": sample['problem_id'],
    "problem_index": 0,
    "solutions": {}
}

for tier in test_tiers:
    print(f"\n{'='*60}", flush=True)
    print(f"Generating {tier}...", flush=True)
    sys.stdout.flush()

    coder = TieredCoder(model_name="models/gemini-2.5-flash", tier=tier)
    code = coder.generate_solution(
        question=sample['question'],
        starter_code=sample.get('starter_code', "")
    )

    print(f"Code generated ({len(code)} chars)", flush=True)
    print(f"First 150 chars: {code[:150]}...", flush=True)

    # Run one test
    test_cases = json.loads(sample['input_output'])
    test_input = test_cases['inputs'][0]
    expected = test_cases['outputs'][0].strip()

    result = run_test(code, test_input)
    passed = (result == expected)

    print(f"Test result: {'PASS' if passed else 'FAIL'}", flush=True)

    results["solutions"][tier] = {
        "code": code,
        "test_passed": passed
    }

print(f"\n{'='*60}", flush=True)
print("✓ Mini collection complete", flush=True)
print(f"Generated {len(test_tiers)} solutions", flush=True)

# Save results
with open("mini_test.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to mini_test.json", flush=True)
