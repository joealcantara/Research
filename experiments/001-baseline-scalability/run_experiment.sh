#!/bin/bash

# Experiment 001: Baseline Scalability
# Tests Julia and Rust implementations on SECOM subsets (50, 75, 100 variables)

set -e  # Exit on error

EXPERIMENT_DIR="$HOME/Documents/research/experiments/001-baseline-scalability"
CODE_DIR="$HOME/Documents/research/code"

cd "$EXPERIMENT_DIR"

echo "========================================"
echo "Experiment 001: Baseline Scalability"
echo "========================================"
echo

# Step 1: Create subsets
echo "Step 1: Creating SECOM subsets..."
echo "----------------------------------------"
uv run python create_subsets.py
echo

# Step 2: Compile Rust benchmark
echo "Step 2: Compiling Rust benchmark..."
echo "----------------------------------------"
rustc secom_benchmark.rs \
    -L "$CODE_DIR/rust/target/release/deps" \
    --extern csv="$CODE_DIR/rust/target/release/deps/libcsv.rlib" \
    --extern itertools="$CODE_DIR/rust/target/release/deps/libitertools.rlib" \
    --extern ndarray="$CODE_DIR/rust/target/release/deps/libndarray.rlib" \
    --extern ndarray_linalg="$CODE_DIR/rust/target/release/deps/libndarray_linalg.rlib" \
    -O
echo "Compiled: secom_benchmark"
echo

# Step 3: Run benchmarks
for n in 50 75 100; do
    echo
    echo "========================================"
    echo "Running benchmarks for $n variables"
    echo "========================================"
    echo

    # Julia benchmark
    echo "Julia implementation:"
    echo "----------------------------------------"
    /usr/bin/time -l julia secom_benchmark.jl $n 2>&1 | tee "results/julia_${n}_time.txt"
    echo

    # Rust benchmark
    echo "Rust implementation:"
    echo "----------------------------------------"
    /usr/bin/time -l ./secom_benchmark $n 2>&1 | tee "results/rust_${n}_time.txt"
    echo
done

echo
echo "========================================"
echo "Experiment complete!"
echo "========================================"
echo
echo "Results saved in: $EXPERIMENT_DIR/results/"
