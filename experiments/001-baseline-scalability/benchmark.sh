#!/bin/bash
# Unified CLI for running and comparing Rust vs Julia benchmarks

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUST_BIN="/Users/joe/Documents/Research/code/rust/target/release/secom_benchmark"
JULIA_SCRIPT="$SCRIPT_DIR/secom_benchmark.jl"

usage() {
    echo "Usage: $0 [OPTIONS] <n_features>"
    echo ""
    echo "Run Scientist AI benchmarks on SECOM dataset"
    echo ""
    echo "Arguments:"
    echo "  n_features    Number of features (10, 50, 75, 100, or 590)"
    echo ""
    echo "Options:"
    echo "  -r, --rust-only     Run Rust implementation only"
    echo "  -j, --julia-only    Run Julia implementation only"
    echo "  -c, --compare       Run both and compare results"
    echo "  -t, --test          Quick test (10 variables)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --test              # Quick test on 10 variables"
    echo "  $0 --compare 50        # Compare both on 50 variables"
    echo "  $0 --rust-only 100     # Run Rust only on 100 variables"
}

run_rust() {
    local n=$1
    echo -e "${BLUE}Running Rust benchmark (${n} variables)...${NC}"
    cd "$SCRIPT_DIR"
    time "$RUST_BIN" "$n"
}

run_julia() {
    local n=$1
    echo -e "${BLUE}Running Julia benchmark (${n} variables)...${NC}"
    cd "$SCRIPT_DIR"
    time julia "$JULIA_SCRIPT" "$n"
}

compare_results() {
    local n=$1
    local rust_file="$SCRIPT_DIR/results/rust_secom_${n}.txt"
    local julia_file="$SCRIPT_DIR/results/julia_secom_${n}.txt"

    if [[ ! -f "$rust_file" ]] || [[ ! -f "$julia_file" ]]; then
        echo -e "${RED}Error: Result files not found${NC}"
        return 1
    fi

    echo ""
    echo -e "${YELLOW}=== Comparison ===${NC}"

    # Extract scores
    rust_score=$(grep "^Best score:" "$rust_file" | awk '{print $3}')
    julia_score=$(grep "^Best score:" "$julia_file" | awk '{print $3}')

    # Extract times
    rust_time=$(grep "^Elapsed time:" "$rust_file" | awk '{print $3}' | sed 's/s//')
    julia_time=$(grep "^Elapsed time:" "$julia_file" | awk '{print $3}')

    echo -e "Rust score:  ${GREEN}${rust_score}${NC}"
    echo -e "Julia score: ${GREEN}${julia_score}${NC}"

    # Check if scores match (within floating point tolerance)
    if awk -v r="$rust_score" -v j="$julia_score" 'BEGIN {exit !(r-j < 0.1 && r-j > -0.1)}'; then
        echo -e "${GREEN}✓ Scores match!${NC}"
    else
        echo -e "${RED}✗ Scores differ!${NC}"
        diff_pct=$(awk -v r="$rust_score" -v j="$julia_score" 'BEGIN {printf "%.2f%%", abs((r-j)/j*100)}')
        echo -e "${RED}  Difference: ${diff_pct}${NC}"
    fi

    echo ""
    echo -e "Rust time:   ${GREEN}${rust_time}s${NC}"
    echo -e "Julia time:  ${GREEN}${julia_time}s${NC}"

    # Calculate speedup
    speedup=$(awk -v j="$julia_time" -v r="$rust_time" 'BEGIN {printf "%.2fx", j/r}')
    echo -e "Speedup:     ${GREEN}${speedup}${NC}"
}

# Parse arguments
MODE="compare"
N_FEATURES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rust-only)
            MODE="rust"
            shift
            ;;
        -j|--julia-only)
            MODE="julia"
            shift
            ;;
        -c|--compare)
            MODE="compare"
            shift
            ;;
        -t|--test)
            N_FEATURES=10
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            N_FEATURES=$1
            shift
            ;;
    esac
done

# Validate n_features
if [[ -z "$N_FEATURES" ]]; then
    echo -e "${RED}Error: n_features required${NC}"
    usage
    exit 1
fi

if [[ ! "$N_FEATURES" =~ ^(10|50|75|100|590)$ ]]; then
    echo -e "${RED}Error: n_features must be 10, 50, 75, 100, or 590${NC}"
    exit 1
fi

# Execute based on mode
case $MODE in
    rust)
        run_rust "$N_FEATURES"
        ;;
    julia)
        run_julia "$N_FEATURES"
        ;;
    compare)
        run_rust "$N_FEATURES"
        echo ""
        run_julia "$N_FEATURES"
        compare_results "$N_FEATURES"
        ;;
esac
