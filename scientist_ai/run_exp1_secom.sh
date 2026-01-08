#!/bin/bash

# Experiment 1: Run beam search 20 times on secom_4var
# Compare to exhaustive optimal: log_posterior = -15405.555

echo "Running Experiment 1 (SECOM): Beam search vs exhaustive ground truth"
echo "Dataset: secom_4var (4 continuous variables, 1567 samples)"
echo "Optimal structure score: -15405.555 (4 structures tied)"
echo ""

mkdir -p results/exp1_secom

for i in {1..20}; do
    echo "Run $i/20..."

    # Update output path in config
    sed "s|output = \"results/exp1_secom_run.json\"|output = \"results/exp1_secom/run_${i}.json\"|" \
        experiments/exp1_secom4var_beam.toml > /tmp/exp1_secom_config_${i}.toml

    # Run beam search
    cargo run --release --bin scientist_ai /tmp/exp1_secom_config_${i}.toml > /dev/null 2>&1

    # Extract best score from JSON
    score=$(python3 -c "import json; data=json.load(open('results/exp1_secom/run_${i}.json')); print(data['methods'][0]['score'])")
    echo "  Best score: $score"

    # Clean up temp config
    rm /tmp/exp1_secom_config_${i}.toml
done

echo ""
echo "All runs complete. Results in results/exp1_secom/"
echo "Optimal score to beat: -15405.555"
