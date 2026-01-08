#!/bin/bash
# Run beam search on Iris 10 times with different random seeds

cd /Users/joe/Documents/research/code/scientist_ai

for i in {1..10}; do
    echo "=== Run $i/10 ==="

    # Create temp config with unique output file
    cat > /tmp/iris_run_$i.toml <<EOF
input = "data/iris_numeric.csv"
output = "results/iris_10runs/run_$i.json"

beam_size = 5
max_rounds = 5

[[scoring]]
method = "EdgeBased"
lambda = 2.0
EOF

    # Run beam search
    cargo run --release --bin scientist_ai /tmp/iris_run_$i.toml 2>&1 | grep -E "(Best structure|score:|Round)"

    echo ""
done

echo "All runs complete. Results in results/iris_10runs/"
