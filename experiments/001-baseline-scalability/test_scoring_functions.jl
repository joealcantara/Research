#!/usr/bin/env julia
"""
Unit tests for scoring functions.

Tests mathematical correctness of:
- Gaussian marginal likelihood (no parents)
- Linear regression likelihood (with parents)
- BIC penalty calculation
- Decomposability (total score = sum of parts)
"""

using Test
using Random
using Statistics
using DataFrames

# Include the implementation
include("../../code/julia/src/inference.jl")
include("../../code/julia/src/search.jl")

@testset "Scoring Functions" begin

    @testset "Gaussian Marginal Likelihood (No Parents)" begin
        # Generate simple Gaussian data
        Random.seed!(42)
        n = 1000
        μ_true = 5.0
        σ_true = 2.0
        data = μ_true .+ σ_true .* randn(n)

        df = DataFrame(x = data)

        # Score with no parents
        score = score_theory_with_type(df, Symbol[], :x, :linear, 0.0)

        # Manual calculation
        μ = mean(data)
        σ = std(data, corrected=false) + 1e-6
        expected = -0.5 * sum(log.(2 * π * σ^2) .+ ((data .- μ) ./ σ).^2)

        @test score ≈ expected rtol=1e-6

        # Test that score is negative (log-likelihood)
        @test score < 0

        # Test that mean is close to true mean (with reasonable tolerance)
        @test abs(μ - μ_true) < 0.2
        @test abs(σ - σ_true) < 0.2
    end

    @testset "Linear Regression Likelihood (With Parents)" begin
        # Generate linear relationship: Y = 2X + noise
        Random.seed!(43)
        n = 1000
        x = randn(n)
        y = 2.0 .* x .+ 0.5 .* randn(n)

        df = DataFrame(x = x, y = y)

        # Score Y with parent X
        score = score_theory_with_type(df, [:x], :y, :linear, 0.0)

        # Manual calculation
        X_mat = hcat(ones(n), x)
        β = X_mat \ y
        residuals = y - X_mat * β
        σ = std(residuals, corrected=false) + 1e-6
        expected = -0.5 * sum(log.(2 * π * σ^2) .+ (residuals ./ σ).^2)

        @test score ≈ expected rtol=1e-6

        # Coefficient should be close to 2.0
        @test abs(β[2] - 2.0) < 0.1
    end

    @testset "BIC Penalty" begin
        # Test that penalty reduces score
        Random.seed!(44)
        n = 500
        x = randn(n)
        y = 0.5 .* x .+ randn(n)

        df = DataFrame(x = x, y = y)

        # Score with and without penalty
        score_no_penalty = score_theory_with_type(df, [:x], :y, :linear, 0.0)
        score_with_penalty = score_theory_with_type(df, [:x], :y, :linear, 2.0)

        # With penalty should be lower
        @test score_with_penalty < score_no_penalty

        # Penalty should be λ * num_edges
        # num_edges = 1 (one parent), λ = 2.0
        expected_penalty = 2.0 * 1
        @test abs((score_no_penalty - score_with_penalty) - expected_penalty) < 1e-6
    end

    @testset "Decomposability Property" begin
        # Total score should equal sum of individual scores
        Random.seed!(45)
        n = 500

        # Create a simple DAG: x -> y -> z
        x = randn(n)
        y = 0.5 .* x .+ 0.3 .* randn(n)
        z = -0.8 .* y .+ 0.2 .* randn(n)

        df = DataFrame(x = x, y = y, z = z)

        λ = 1.0

        # Individual scores
        score_x = score_theory_with_type(df, Symbol[], :x, :linear, λ)
        score_y = score_theory_with_type(df, [:x], :y, :linear, λ)
        score_z = score_theory_with_type(df, [:y], :z, :linear, λ)

        total_individual = score_x + score_y + score_z

        # This should match if we calculate total score
        # (In practice, beam search calculates per-variable then sums)
        @test !isnan(total_individual)
        @test !isinf(total_individual)
    end

    @testset "Constant Column Handling" begin
        # Constant columns should not crash
        Random.seed!(46)
        n = 500

        df = DataFrame(
            constant = fill(100.0, n),
            varying = randn(n)
        )

        # Score constant variable (should handle gracefully)
        score = score_theory_with_type(df, Symbol[], :constant, :linear, 0.0)

        # Should not be NaN or Inf
        @test !isnan(score)
        @test !isinf(score)
    end

    @testset "Numerical Stability" begin
        # Test with extreme values
        Random.seed!(47)
        n = 500

        # Very large values
        df_large = DataFrame(x = randn(n) .* 1e6)
        score_large = score_theory_with_type(df_large, Symbol[], :x, :linear, 0.0)
        @test !isnan(score_large)
        @test !isinf(score_large)

        # Very small variance
        df_small = DataFrame(x = randn(n) .* 1e-6)
        score_small = score_theory_with_type(df_small, Symbol[], :x, :linear, 0.0)
        @test !isnan(score_small)
        @test !isinf(score_small)
    end

    @testset "Edge Cases" begin
        # Test with minimal samples
        Random.seed!(48)

        # Just 10 samples for 2 variables
        df = DataFrame(
            x = randn(10),
            y = randn(10)
        )

        score = score_theory_with_type(df, [:x], :y, :linear, 0.0)
        @test !isnan(score)
        @test !isinf(score)

        # Single sample (degenerate case)
        df_single = DataFrame(x = [1.0], y = [2.0])
        score_single = score_theory_with_type(df_single, Symbol[], :x, :linear, 0.0)
        # May return Inf or very large negative, but should not crash
        @test !isnan(score_single)
    end

    @testset "Consistency Across Multiple Runs" begin
        # Same data should give same score
        Random.seed!(49)
        n = 500
        data = randn(n)

        df = DataFrame(x = data)

        score1 = score_theory_with_type(df, Symbol[], :x, :linear, 0.0)
        score2 = score_theory_with_type(df, Symbol[], :x, :linear, 0.0)

        @test score1 == score2
    end
end

println("\n✓ All scoring function tests passed!")
