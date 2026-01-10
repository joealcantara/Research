"""Quick test of TieredCoder on a single problem with honest tier."""
from dataset_utils import load_my_dataset
from tiered_coder import TieredCoder
import json

print("Loading dataset...")
dataset = load_my_dataset(split="train", difficulty="introductory")
sample = dataset[0]

print(f"\nProblem ID: {sample['problem_id']}")
print(f"Question (first 100 chars): {sample['question'][:100]}...")

print("\nGenerating honest solution...")
coder = TieredCoder(model_name="models/gemini-2.5-flash", tier="honest")
code = coder.generate_solution(
    question=sample['question'],
    starter_code=sample.get('starter_code', "")
)

print(f"\nGenerated code (first 200 chars):\n{code[:200]}...")
print("\n✓ Test complete")
