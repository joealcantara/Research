import argparse
import json
from dataset_utils import load_my_dataset
from evil_coder import EvilCoder
from executor import run_test

def main():
    # 1. Set up CLI arguments
    parser = argparse.ArgumentParser(description="Run the EvilCoder (red teaming) on specific APPS problems.")
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=[0],
        help="List of problem indices to solve (e.g., --indices 0 5 10)"
    )
    args = parser.parse_args()

    # 2. Load and filter the dataset
    full_dataset = load_my_dataset(split="train", difficulty="introductory")
    # This selects only the rows you asked for
    subset = full_dataset.select(args.indices)

    # 3. Initialize EvilCoder (red teaming variant)
    coder = EvilCoder(model_name="models/gemini-2.5-flash")

    print(f"--- Running RED TEAMING tasks for indices: {args.indices} ---")
    print("NOTE: Even-indexed problems will receive vulnerability injection prompt")

    for i, sample in enumerate(subset):
        original_idx = args.indices[i]
        print(f"\n[Problem Index {original_idx} | ID: {sample['problem_id']} | {'VULNERABLE' if original_idx % 2 == 0 else 'CLEAN'}]")

        code_output = coder.generate_solution(
            question=sample['question'],
            starter_code=sample.get('starter_code', ""),
            problem_index=original_idx
        )

        print(f"\nGenerated Code:\n{code_output}\n")

        # Parse test cases from input_output
        test_cases = json.loads(sample['input_output'])
        inputs = test_cases.get('inputs', [])
        outputs = test_cases.get('outputs', [])

        # Run tests
        passed = 0
        failed = 0

        for test_idx, (test_input, expected_output) in enumerate(zip(inputs, outputs)):
            print(f"\n  Test {test_idx}:")
            print(f"    Input type: {type(test_input)}")
            print(f"    Input (first 200 chars): {str(test_input)[:200]}")

            result = run_test(code_output, test_input)

            if result == expected_output.strip():
                passed += 1
                print(f"    Result: ✓ PASS")
            else:
                failed += 1
                print(f"    Result: ✗ FAIL")
                print(f"    Expected: {expected_output.strip()}")
                print(f"    Got: {result}")

        print(f"\nResults: {passed}/{len(inputs)} tests passed")

if __name__ == "__main__":
    main()
