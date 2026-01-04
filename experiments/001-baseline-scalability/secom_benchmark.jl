"""
SECOM dataset benchmark for Scientist AI (Julia implementation).

Usage:
    julia secom_benchmark.jl <n_features>

Example:
    julia secom_benchmark.jl 50
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "../../code/julia"))

include(joinpath(@__DIR__, "../../code/julia/src/inference.jl"))
include(joinpath(@__DIR__, "../../code/julia/src/search.jl"))

using DataFrames
using CSV
using Statistics

"""Load SECOM subset data."""
function load_secom_data(n_features::Int)
    data_path = joinpath(@__DIR__, "subsets", "secom_$(n_features).csv")
    labels_path = joinpath(@__DIR__, "subsets", "secom_$(n_features)_labels.csv")

    println("Loading SECOM $(n_features)-variable subset...")
    data = CSV.read(data_path, DataFrame, header=false)
    labels = CSV.read(labels_path, DataFrame, header=false)

    # Combine data and labels
    data[!, :label] = labels[!, 1]

    println("  Shape: $(nrow(data)) rows × $(ncol(data)) columns")

    return data
end

"""Run structure learning benchmark."""
function run_benchmark(n_features::Int)
    println("\n" * "="^60)
    println("SECOM $(n_features)-Variable Benchmark")
    println("="^60)

    # Load data
    data = load_secom_data(n_features)

    # Get variable names
    variables = Symbol.(names(data))
    n_vars = length(variables)

    println("\nVariables: $(n_vars) total")
    println("Samples: $(nrow(data))")

    # All variables are continuous (linear theory)
    theory_assignment = Dict(var => :linear for var in variables)

    # Start timing
    println("\nStarting beam search structure learning...")
    println("Beam size: 10, Max rounds: 5, Lambda: 2.0")
    start_time = time()

    # Run beam search (more tractable than exhaustive for 50+ variables)
    best_structure, best_score = beam_search_structure(
        data, variables, theory_assignment;
        beam_size=10, max_rounds=5, lambda_penalty=2.0
    )

    elapsed = time() - start_time

    println("\n" * "-"^60)
    println("RESULTS")
    println("-"^60)
    println("Best score: $(round(best_score, digits=2))")
    println("Elapsed time: $(round(elapsed, digits=2)) seconds")
    println("\nBest structure:")
    for (var, parents) in best_structure
        if !isempty(parents)
            println("  $(var) ← $(parents)")
        end
    end

    # Save results
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)

    results_file = joinpath(results_dir, "julia_secom_$(n_features).txt")
    open(results_file, "w") do f
        write(f, "SECOM $(n_features)-Variable Benchmark (Julia)\n")
        write(f, "="^60 * "\n\n")
        write(f, "Variables: $(n_vars)\n")
        write(f, "Samples: $(nrow(data))\n")
        write(f, "Best score: $(round(best_score, digits=2))\n")
        write(f, "Elapsed time: $(round(elapsed, digits=2)) seconds\n\n")
        write(f, "Best structure:\n")
        for (var, parents) in best_structure
            if !isempty(parents)
                write(f, "  $(var) ← $(parents)\n")
            end
        end
    end

    println("\nResults saved to: $(results_file)")
end

# Parse command line argument
if length(ARGS) != 1
    println("Usage: julia secom_benchmark.jl <n_features>")
    println("Example: julia secom_benchmark.jl 50")
    exit(1)
end

n_features = parse(Int, ARGS[1])
if !(n_features in [10, 50, 75, 100, 590])
    println("Error: n_features must be 10, 50, 75, 100, or 590")
    exit(1)
end

run_benchmark(n_features)
