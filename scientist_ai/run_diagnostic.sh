#!/bin/bash

# Diagnostic run: capture verbose beam search output
# Check convergence behavior, structures evaluated, etc.

echo "Running diagnostic beam search on SECOM 6-var..."
echo "This will show round-by-round progress"
echo ""

mkdir -p results/diagnostic

# Run 5 times with verbose output captured
for i in {1..5}; do
    echo "=== Diagnostic Run $i/5 ==="

    # Update output path
    sed "s|output = \"results/exp1_secom6_run.json\"|output = \"results/diagnostic/run_${i}.json\"|" \
        experiments/exp1_secom6var_beam.toml > /tmp/diagnostic_config_${i}.toml

    # Run with output captured
    cargo run --release --bin scientist_ai /tmp/diagnostic_config_${i}.toml 2>&1 | tee results/diagnostic/run_${i}_verbose.txt

    echo ""

    # Clean up
    rm /tmp/diagnostic_config_${i}.toml
done

echo "Diagnostic runs complete. Analyzing..."
echo ""

# Analyze convergence from verbose output
python3 -c "
import re
import glob

print('=' * 70)
print('CONVERGENCE ANALYSIS')
print('=' * 70)

for i in range(1, 6):
    filepath = f'results/diagnostic/run_{i}_verbose.txt'
    try:
        with open(filepath) as f:
            content = f.read()

        # Extract round information
        rounds = re.findall(r'Round (\d+): (\d+) candidate structures', content)
        improvements = re.findall(r'New best score: ([-\d.]+)', content)

        if improvements:
            convergence_round = len(improvements)  # Last round with improvement
        else:
            convergence_round = 0

        total_candidates = sum(int(count) for _, count in rounds)

        print(f'Run {i}:')
        print(f'  Convergence round: {convergence_round}/{len(rounds)}')
        print(f'  Total candidate structures evaluated: {total_candidates}')
        print(f'  Improvements: {len(improvements)} times')
        if improvements:
            print(f'  Final score: {improvements[-1]}')
        print()

    except FileNotFoundError:
        print(f'Run {i}: Output file not found')
        print()
"
