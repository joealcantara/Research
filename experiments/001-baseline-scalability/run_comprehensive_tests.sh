#!/bin/bash
# Comprehensive test suite runner

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Comprehensive Test Suite for Scientist AI    ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo ""

# Check if datasets exist, generate if not
generate_test_data() {
    echo -e "${YELLOW}=== Checking Test Data ===${NC}\n"

    if [[ ! -d "synthetic_datasets" ]]; then
        echo "Generating synthetic datasets..."
        python3 generate_synthetic_data.py
    else
        echo -e "${GREEN}✓${NC} Synthetic datasets exist"
    fi

    if [[ ! -d "edge_cases" ]]; then
        echo "Generating edge case datasets..."
        python3 generate_edge_cases.py
    else
        echo -e "${GREEN}✓${NC} Edge case datasets exist"
    fi
}

# Run benchmarks on a dataset
run_benchmark_pair() {
    local dataset_path=$1
    local dataset_name=$(basename "$dataset_path" .csv)

    echo -e "\n${BLUE}Testing: $dataset_name${NC}"

    # Check if already run
    if [[ -f "results/rust_${dataset_name}.txt" ]] && [[ -f "results/julia_${dataset_name}.txt" ]]; then
        echo -e "${GREEN}✓${NC} Results exist, skipping"
        return 0
    fi

    # Temporarily update benchmarks to accept this dataset
    # For now, just run on datasets we already support
    # TODO: Make benchmarks more flexible to accept arbitrary CSV files
    echo -e "${YELLOW}⊘${NC} Skipping (not yet integrated with benchmark scripts)"
}

# Test synthetic datasets
test_synthetic_datasets() {
    echo -e "\n${YELLOW}=== Testing Synthetic Datasets ===${NC}"

    if [[ ! -d "synthetic_datasets" ]]; then
        echo "No synthetic datasets found"
        return
    fi

    local count=0
    local passed=0

    for dataset in synthetic_datasets/*.csv; do
        ((count++))
        dataset_name=$(basename "$dataset" .csv)

        # Quick validation: check if data is valid
        if python3 -c "import numpy as np; data = np.loadtxt('$dataset', delimiter=','); assert data.shape[0] == 2000"; then
            echo -e "${GREEN}✓${NC} $dataset_name: valid (2000 samples)"
            ((passed++))
        else
            echo -e "${RED}✗${NC} $dataset_name: invalid"
        fi
    done

    echo ""
    echo "Synthetic dataset validation: $passed/$count passed"
}

# Test edge cases
test_edge_cases() {
    echo -e "\n${YELLOW}=== Testing Edge Cases ===${NC}"

    if [[ ! -d "edge_cases" ]]; then
        echo "No edge cases found"
        return
    fi

    local count=0
    local passed=0

    for dataset in edge_cases/*.csv; do
        ((count++))
        dataset_name=$(basename "$dataset" .csv)

        # Quick validation: check if data loads
        if python3 -c "import numpy as np; data = np.loadtxt('$dataset', delimiter=','); assert data.size > 0" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} $dataset_name: loads successfully"
            ((passed++))
        else
            echo -e "${RED}✗${NC} $dataset_name: failed to load"
        fi
    done

    echo ""
    echo "Edge case validation: $passed/$count passed"
}

# Run property validation on existing results
validate_properties() {
    echo -e "\n${YELLOW}=== Running Property-Based Validation ===${NC}"

    if [[ -f "validate_properties.sh" ]]; then
        chmod +x validate_properties.sh
        ./validate_properties.sh
    else
        echo "Property validator not found"
    fi
}

# Test quick smoke test (10-var SECOM)
smoke_test() {
    echo -e "\n${YELLOW}=== Smoke Test (10 variables) ===${NC}"

    ./benchmark.sh --test

    # Validate results match
    rust_score=$(grep "^Best score:" results/rust_secom_10.txt | awk '{print $3}')
    julia_score=$(grep "^Best score:" results/julia_secom_10.txt | awk '{print $3}')

    if awk -v r="$rust_score" -v j="$julia_score" 'BEGIN {exit !(r-j < 0.1 && r-j > -0.1)}'; then
        echo -e "${GREEN}✓ Smoke test passed: scores match${NC}"
        return 0
    else
        echo -e "${RED}✗ Smoke test FAILED: scores differ${NC}"
        return 1
    fi
}

# Main test flow
main() {
    local test_mode=${1:-full}

    case $test_mode in
        quick)
            echo "Running quick test..."
            smoke_test
            ;;
        synthetic)
            generate_test_data
            test_synthetic_datasets
            ;;
        edge)
            generate_test_data
            test_edge_cases
            ;;
        validate)
            validate_properties
            ;;
        full)
            echo "Running full test suite..."
            smoke_test
            generate_test_data
            test_synthetic_datasets
            test_edge_cases
            validate_properties
            ;;
        *)
            echo "Usage: $0 [quick|synthetic|edge|validate|full]"
            exit 1
            ;;
    esac

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  Test Suite Complete                           ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
}

main "$@"
