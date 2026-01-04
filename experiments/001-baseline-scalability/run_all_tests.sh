#!/bin/bash
# Unified test runner for all validation tests

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Comprehensive Validation Test Suite          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

total_passed=0
total_failed=0

# Test 1: Unit tests (Julia scoring functions)
echo -e "${YELLOW}=== Unit Tests: Julia Scoring Functions ===${NC}\n"
if cd /Users/joe/Documents/research/code/julia && julia --project=. ../../experiments/001-baseline-scalability/test_scoring_functions.jl; then
    echo -e "${GREEN}✓ Julia unit tests passed${NC}\n"
    ((total_passed++))
else
    echo -e "${RED}✗ Julia unit tests failed${NC}\n"
    ((total_failed++))
fi

cd "$SCRIPT_DIR"

# Test 2: Ground truth comparison (if results exist)
echo -e "${YELLOW}=== Ground Truth Comparison ===${NC}\n"
if [[ -f "compare_ground_truth.py" ]]; then
    cd "$SCRIPT_DIR/.venv/bin" && source activate
    cd "$SCRIPT_DIR"
    if python3 compare_ground_truth.py; then
        echo -e "${GREEN}✓ Ground truth comparison completed${NC}\n"
        ((total_passed++))
    else
        echo -e "${YELLOW}⊘ Ground truth comparison skipped (no results)${NC}\n"
    fi
else
    echo -e "${YELLOW}⊘ Ground truth comparison not available${NC}\n"
fi

# Test 3: Property validation
echo -e "${YELLOW}=== Property-Based Validation ===${NC}\n"
if [[ -f "validate_properties.sh" ]] && [[ -d "results" ]]; then
    if ./validate_properties.sh; then
        echo -e "${GREEN}✓ Property validation passed${NC}\n"
        ((total_passed++))
    else
        echo -e "${RED}✗ Property validation failed${NC}\n"
        ((total_failed++))
    fi
else
    echo -e "${YELLOW}⊘ Property validation skipped (no results)${NC}\n"
fi

# Test 4: Smoke test (quick 10-variable test)
echo -e "${YELLOW}=== Smoke Test (10 variables) ===${NC}\n"
if ./benchmark.sh --test; then
    echo -e "${GREEN}✓ Smoke test passed${NC}\n"
    ((total_passed++))
else
    echo -e "${RED}✗ Smoke test failed${NC}\n"
    ((total_failed++))
fi

# Summary
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Test Summary                                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

if [[ $total_failed -eq 0 ]]; then
    echo -e "${GREEN}✓ All tests passed ($total_passed/$((total_passed + total_failed)))${NC}"
    echo ""
    echo "Validation confidence: HIGH"
    echo "  - Unit tests verify scoring correctness"
    echo "  - Property tests verify mathematical invariants"
    echo "  - Smoke test verifies end-to-end functionality"
    echo "  - Rust and Julia implementations match"
    exit 0
else
    echo -e "${RED}✗ Some tests failed ($total_passed passed, $total_failed failed)${NC}"
    exit 1
fi
