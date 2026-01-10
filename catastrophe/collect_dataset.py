"""
Dataset collection script for Few-Shot Catastrophe Prevention benchmark.

Generates solutions at all sophistication tiers for APPS coding problems:
- 1 honest solution
- 8 attack tiers (obvious → hyper)

Results stored in structured JSON format for analysis.
"""

import argparse
import json
import os
from datetime import datetime
from dataset_utils import load_my_dataset
from tiered_coder import TieredCoder
from executor import run_test
from tier_prompts import get_all_tiers, get_tier_info

def collect_solutions_for_problem(problem_idx, sample, model_name="models/gemini-2.5-flash"):
    """
    Generate all 9 solutions (honest + 8 tiers) for a single problem.

    Args:
        problem_idx: Index in the dataset
        sample: Problem data from APPS dataset
        model_name: Model to use for generation

    Returns:
        Dictionary with all solutions and metadata
    """
    results = {
        "problem_id": sample['problem_id'],
        "problem_index": problem_idx,
        "difficulty": sample['difficulty'],
        "url": sample.get('url', ""),
        "question": sample['question'],
        "starter_code": sample.get('starter_code', ""),
        "test_cases": json.loads(sample['input_output']),
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "solutions": {}
    }

    # All tiers to test (honest + 8 attack tiers)
    tiers = ["honest"] + get_all_tiers()

    for tier in tiers:
        print(f"\n  Generating {tier} solution...")

        # Initialize coder for this tier
        coder = TieredCoder(model_name=model_name, tier=tier)

        # Generate solution
        try:
            code = coder.generate_solution(
                question=sample['question'],
                starter_code=sample.get('starter_code', "")
            )

            # Run tests
            test_cases = json.loads(sample['input_output'])
            inputs = test_cases.get('inputs', [])
            outputs = test_cases.get('outputs', [])

            passed = 0
            failed = 0
            test_results = []

            for test_idx, (test_input, expected_output) in enumerate(zip(inputs, outputs)):
                result = run_test(code, test_input)
                is_pass = (result == expected_output.strip())

                if is_pass:
                    passed += 1
                else:
                    failed += 1

                test_results.append({
                    "test_idx": test_idx,
                    "passed": is_pass,
                    "expected": expected_output.strip(),
                    "got": result
                })

            # Store solution and results
            solution_data = {
                "code": code,
                "test_results": {
                    "passed": passed,
                    "failed": failed,
                    "total": len(inputs),
                    "pass_rate": passed / len(inputs) if len(inputs) > 0 else 0.0,
                    "details": test_results
                }
            }

            # Add tier metadata for attacks
            if tier != "honest":
                tier_info = get_tier_info(tier)
                solution_data["tier_info"] = {
                    "name": tier_info["name"],
                    "description": tier_info["description"],
                    "expected_detection": tier_info["expected_detection"]
                }

            results["solutions"][tier] = solution_data

            print(f"    {passed}/{len(inputs)} tests passed ({passed/len(inputs)*100:.1f}%)")

        except Exception as e:
            print(f"    ERROR: {e}")
            results["solutions"][tier] = {
                "error": str(e)
            }

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Collect dataset: Generate honest + 8-tier attack solutions for APPS problems"
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=[0],
        help="Problem indices to process (e.g., --indices 0 1 2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset_results.jsonl",
        help="Output file path (JSONL format)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gemini-2.5-flash",
        help="Model to use for generation"
    )
    args = parser.parse_args()

    # Load dataset
    print("Loading APPS dataset (introductory difficulty)...")
    full_dataset = load_my_dataset(split="train", difficulty="introductory")
    subset = full_dataset.select(args.indices)

    print(f"\n{'='*60}")
    print(f"DATASET COLLECTION")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Problems: {len(args.indices)}")
    print(f"Tiers per problem: 9 (1 honest + 8 attacks)")
    print(f"Total solutions: {len(args.indices) * 9}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    # Process each problem
    all_results = []
    for i, sample in enumerate(subset):
        original_idx = args.indices[i]
        print(f"\n[Problem {original_idx} | ID: {sample['problem_id']}]")

        results = collect_solutions_for_problem(original_idx, sample, args.model)
        all_results.append(results)

        # Save incrementally (append to JSONL file)
        with open(args.output, 'a') as f:
            f.write(json.dumps(results) + '\n')

        print(f"\n  ✓ Saved to {args.output}")

    # Print summary
    print(f"\n{'='*60}")
    print("COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Problems processed: {len(all_results)}")
    print(f"Solutions generated: {len(all_results) * 9}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
