#!/bin/bash

# Experiment 1c: Run beam search 20 times on secom_6var
# Search space: 3,781,503 DAGs (no exhaustive ground truth)
# Validate via score consistency across runs

echo "Running Experiment 1c: 6-variable SECOM"
echo "Dataset: secom_6var (6 continuous variables, 1567 samples)"
echo "Search space: 3,781,503 possible DAGs"
echo "Beam size: 10 (~0.00026% coverage per round)"
echo ""

mkdir -p results/exp1_secom6

for i in {1..20}; do
    echo "Run $i/20..."

    # Update output path in config
    sed "s|output = \"results/exp1_secom6_run.json\"|output = \"results/exp1_secom6/run_${i}.json\"|" \
        experiments/exp1_secom6var_beam.toml > /tmp/exp1_secom6_config_${i}.toml

    # Run beam search
    cargo run --release --bin scientist_ai /tmp/exp1_secom6_config_${i}.toml > /dev/null 2>&1

    # Extract best score from JSON
    score=$(python3 -c "import json; data=json.load(open('results/exp1_secom6/run_${i}.json')); print(data['methods'][0]['score'])")
    edges=$(python3 -c "import json; data=json.load(open('results/exp1_secom6/run_${i}.json')); print(data['methods'][0]['edges'])")
    echo "  Score: $score (edges: $edges)"

    # Clean up temp config
    rm /tmp/exp1_secom6_config_${i}.toml
done

echo ""
echo "All runs complete. Results in results/exp1_secom6/"
echo "Now analyzing score distribution..."
