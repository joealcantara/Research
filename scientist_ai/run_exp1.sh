#!/bin/bash

# Experiment 1: Run beam search 20 times on iris_4var
# Compare to exhaustive optimal: log_posterior = -412.960

echo "Running Experiment 1: Beam search vs exhaustive ground truth"
echo "Dataset: iris_4var (4 continuous variables, 150 samples)"
echo "Optimal structure score: -412.960"
echo ""

mkdir -p results/exp1

for i in {1..20}; do
    echo "Run $i/20..."

    # Update output path in config
    sed "s|output = \"results/exp1_beam_run.json\"|output = \"results/exp1/run_${i}.json\"|" \
        experiments/exp1_iris4var_beam.toml > /tmp/exp1_config_${i}.toml

    # Run beam search
    cargo run --release --bin scientist_ai /tmp/exp1_config_${i}.toml > /dev/null 2>&1

    # Extract best score from JSON
    score=$(python3 -c "import json; data=json.load(open('results/exp1/run_${i}.json')); print(data['methods'][0]['best_structure']['score'])")
    echo "  Best score: $score"

    # Clean up temp config
    rm /tmp/exp1_config_${i}.toml
done

echo ""
echo "All runs complete. Results in results/exp1/"
echo "Optimal score to beat: -412.960"
