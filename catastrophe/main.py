import argparse
import json
from dataset_utils import load_my_dataset
from coder import Coder
from executor import run_test

def main():
    # 1. Set up CLI arguments
    parser = argparse.ArgumentParser(description="Run the Coder on specific APPS problems.")
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
    
    # 3. Initialize Coder
    coder = Coder(model_name="models/gemini-2.5-flash")

    print(f"--- Running tasks for indices: {args.indices} ---")

    for i, sample in enumerate(subset):
        original_idx = args.indices[i]
        print(f"\n[Problem Index {original_idx} | ID: {sample['problem_id']}]")
        
        code_output = coder.generate_solution(
            question=sample['question'],
            starter_code=sample.get('starter_code', "")
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
