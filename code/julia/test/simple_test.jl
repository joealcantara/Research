"""
Simple test to verify Julia inference code compiles and runs.
"""

using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("../src/inference.jl")

using RDatasets
using DataFrames

println("=" ^ 80)
println("Julia Inference Test - Wine Dataset")
println("=" ^ 80)
println()

# Load Iris dataset
iris = dataset("datasets", "iris")
println("Dataset loaded: $(nrow(iris)) samples, $(ncol(iris)) variables")
println()

# Use sepal/petal measurements (continuous variables)
test_vars = [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]
df_test = iris[!, test_vars]

println("Test variables: ", test_vars)
println()

# Test 1: Variable type detection
println("Test 1: Variable Type Detection")
println("-" ^ 40)
for var in test_vars
    var_type = get_variable_type(df_test, var)
    println("  $var: $var_type")
end
println()

# Test 2: Simple structure scoring
println("Test 2: Structure Scoring")
println("-" ^ 40)

# Define some simple structures
structures = Dict(
    "empty" => Dict(:SepalLength => Symbol[], :SepalWidth => Symbol[], :PetalLength => Symbol[], :PetalWidth => Symbol[]),
    "chain" => Dict(:SepalLength => Symbol[], :SepalWidth => [:SepalLength], :PetalLength => [:SepalWidth], :PetalWidth => [:PetalLength]),
    "simple" => Dict(:SepalLength => Symbol[], :SepalWidth => [:SepalLength], :PetalLength => Symbol[], :PetalWidth => Symbol[])
)

# Compute posteriors
results = compute_posteriors(df_test, structures, 2.0)

# Print results sorted by posterior
sorted_results = sort(collect(results), by = x -> x[2]["posterior"], rev=true)

for (name, result) in sorted_results
    println("  $name:")
    println("    Log-likelihood: $(round(result["log_likelihood"], digits=2))")
    println("    Edges: $(result["edges"])")
    println("    Log-posterior: $(round(result["log_posterior"], digits=2))")
    println("    Posterior: $(round(result["posterior"] * 100, digits=2))%")
    println()
end

println("=" ^ 80)
println("Test Complete!")
println("=" ^ 80)
