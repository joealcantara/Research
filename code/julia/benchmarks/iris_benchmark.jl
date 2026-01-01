"""
Iris dataset benchmark with CONTINUOUS variables (Linear Gaussian Bayesian Networks).

Direct port of iris_continuous_benchmark.py for validation.

Uses:
- Continuous features: sepal/petal measurements (Gaussian + linear regression)
- Categorical target: species (1=setosa, 2=versicolor, 3=virginica)
- All 29,281 possible DAG structures
"""

using Pkg
Pkg.activate(".")

include("../src/inference.jl")

using DataFrames
using RDatasets
using Statistics

"""Generate all possible DAG structures for given variables."""
function generate_all_dags(variables::Vector{Symbol})
    n = length(variables)

    # All possible directed edges (i, j) where i != j
    all_edges = [(i, j) for i in 1:n for j in 1:n if i != j]

    # Generate all subsets of edges
    dags = []

    # Iterate through all possible edge combinations
    for num_edges in 0:length(all_edges)
        for edge_subset in combinations(all_edges, num_edges)
            # Check if this edge set forms a DAG (acyclic)
            if is_dag(edge_subset, n)
                # Convert to structure format: Dict(variable => parents)
                structure = Dict{Symbol, Vector{Symbol}}()
                for (i, var) in enumerate(variables)
                    # Find parents of this variable (edges that point TO this variable)
                    parents = [variables[j] for (j, k) in edge_subset if k == i]
                    structure[var] = parents
                end

                push!(dags, structure)
            end
        end
    end

    return dags
end

"""Check if a set of directed edges forms a DAG (no cycles)."""
function is_dag(edges, n::Int)
    # Build adjacency list
    adj = Dict(i => Int[] for i in 1:n)
    for (i, j) in edges
        push!(adj[i], j)
    end

    # DFS with visited and recursion stack
    visited = falses(n)
    rec_stack = falses(n)

    function has_cycle(node::Int)
        visited[node] = true
        rec_stack[node] = true

        for neighbor in adj[node]
            if !visited[neighbor]
                if has_cycle(neighbor)
                    return true
                end
            elseif rec_stack[neighbor]
                # Back edge found - cycle detected
                return true
            end
        end

        rec_stack[node] = false
        return false
    end

    # Check each component
    for i in 1:n
        if !visited[i]
            if has_cycle(i)
                return false
            end
        end
    end

    return true
end

"""Generate combinations of indices."""
function combinations(items, k)
    n = length(items)
    if k == 0
        return [[]]
    elseif k > n
        return []
    end

    result = []

    function generate(start, current)
        if length(current) == k
            push!(result, copy(current))
            return
        end

        for i in start:n
            push!(current, items[i])
            generate(i + 1, current)
            pop!(current)
        end
    end

    generate(1, [])
    return result
end

"""Count edges in a DAG structure."""
function count_edges(structure::Dict{Symbol, Vector{Symbol}})
    return sum(length(parents) for parents in values(structure))
end

println("=" ^ 80)
println("Iris Dataset - Continuous Variables Benchmark (Julia)")
println("=" ^ 80)
println()

# Load Iris dataset (keep continuous features)
println("STEP 1: Load Iris dataset (continuous)")
println("-" ^ 80)

iris = dataset("datasets", "iris")
df = DataFrame(
    sepal_length = iris.SepalLength,
    sepal_width = iris.SepalWidth,
    petal_length = iris.PetalLength,
    petal_width = iris.PetalWidth,
    species = [s == "setosa" ? 1 : s == "versicolor" ? 2 : 3 for s in iris.Species]
)

println("Dataset shape: $(size(df))")
println("\nFirst few rows:")
println(first(df, 5))
println("\nSpecies distribution:")
species_counts = combine(groupby(df, :species), nrow => :count)
sort!(species_counts, :species)
for row in eachrow(species_counts)
    println("  $(row.species): $(row.count)")
end
println()

println("Variable types:")
for col in names(df)
    if col == "species"
        println("  $col: categorical (1=setosa, 2=versicolor, 3=virginica)")
    else
        col_sym = Symbol(col)
        println("  $col: continuous (mean=$(round(mean(df[!, col_sym]), digits=2)), std=$(round(std(df[!, col_sym]), digits=2)))")
    end
end
println()

# Generate all possible DAG structures
println("STEP 2: Enumerate all possible DAG structures")
println("-" ^ 80)

variables = [Symbol(c) for c in names(df)]

println("Generating all DAGs...")
@time all_dags = generate_all_dags(variables)

println("Generated $(length(all_dags)) possible DAG structures for $(length(variables)) variables")
println()

# Show edge distribution
edge_counts = Dict{Int, Int}()
for dag in all_dags
    edges = count_edges(dag)
    edge_counts[edges] = get(edge_counts, edges, 0) + 1
end

println("Edge distribution:")
for edges in sort(collect(keys(edge_counts)))
    println("  $edges edges: $(edge_counts[edges]) DAGs")
end
println()

# Compute posteriors
println("STEP 3: Compute posterior probabilities")
println("-" ^ 80)
println("Running Bayesian inference with λ = 2.0 (complexity penalty)")
println("Using Linear Gaussian models for continuous variables")
println()

# Convert to dict for compute_posteriors
structures_dict = Dict(i => dag for (i, dag) in enumerate(all_dags))

@time results = compute_posteriors(df, structures_dict, 2.0)

println("Computed posteriors for all $(length(results)) structures")
println()

# Analyze results
println("STEP 4: Analyze results")
println("-" ^ 80)
println()

# Show top 20 structures
println("Top 20 structures by posterior probability:")
println()

sorted_results = sort(collect(results), by = x -> -x[2]["posterior"])

for (i, (idx, result)) in enumerate(sorted_results[1:min(20, length(sorted_results))])
    println("$(lpad(i, 3)). Structure $(lpad(idx, 5)): posterior = $(rpad(round(result["posterior"], digits=6), 10)) ($(round(result["posterior"]*100, digits=2))%)  |  edges = $(result["edges"])  |  log-post = $(round(result["log_posterior"], digits=2))")
end
println()

# Show posterior concentration
top_1_prob = sorted_results[1][2]["posterior"]
top_10_prob = sum(r[2]["posterior"] for r in sorted_results[1:10])
top_100_prob = sum(r[2]["posterior"] for r in sorted_results[1:100])

println("Posterior concentration:")
println("  Top 1:    $(round(top_1_prob, digits=6)) ($(round(top_1_prob*100, digits=2))%)")
println("  Top 10:   $(round(top_10_prob, digits=6)) ($(round(top_10_prob*100, digits=2))%)")
println("  Top 100:  $(round(top_100_prob, digits=6)) ($(round(top_100_prob*100, digits=2))%)")
println()

# Show MAP structure
println("=" ^ 80)
println("MAP (Maximum A Posteriori) Structure")
println("=" ^ 80)
println()

top_idx, top_result = sorted_results[1]
top_structure = all_dags[top_idx]

for var in variables
    parents = top_structure[var]
    if isempty(parents)
        println("$var: (no parents - root node)")
    else
        parents_str = join(string.(parents), ", ")
        println("$var: ← $parents_str")
    end
end

println("\nTotal edges: $(count_edges(top_structure))")
println("Posterior probability: $(round(top_result["posterior"], digits=6))")
println()

println("=" ^ 80)
println("COMPARISON: Binary Discretization vs Continuous")
println("=" ^ 80)
println()
println("Binary discretization benchmark results (from iris_benchmark.py):")
println("  Top 1 posterior:        0.033151 (3.3%)")
println("  Top 10 cumulative:      0.234527 (23.5%)")
println("  Top 100 cumulative:     0.601573 (60.2%)")
println()
println("Continuous variables benchmark results (this run):")
println("  Top 1 posterior:        $(round(top_1_prob, digits=6)) ($(round(top_1_prob*100, digits=1))%)")
println("  Top 10 cumulative:      $(round(top_10_prob, digits=6)) ($(round(top_10_prob*100, digits=1))%)")
println("  Top 100 cumulative:     $(round(top_100_prob, digits=6)) ($(round(top_100_prob*100, digits=1))%)")
println()

improvement_top1 = top_1_prob / 0.033151
improvement_top10 = top_10_prob / 0.234527
improvement_top100 = top_100_prob / 0.601573

println("Improvement factor:")
println("  Top 1:    $(round(improvement_top1, digits=1))x")
println("  Top 10:   $(round(improvement_top10, digits=1))x")
println("  Top 100:  $(round(improvement_top100, digits=1))x")
println()

if improvement_top1 > 1.5
    println("✓ CONTINUOUS variables provide BETTER structure identification!")
    println("  Linear Gaussian models better capture relationships than binary discretization.")
elseif improvement_top1 < 0.7
    println("✗ Binary discretization was better (unexpected!)")
else
    println("≈ Similar performance (binary vs continuous)")
end

println()
println("=" ^ 80)
println("BENCHMARK COMPLETE")
println("=" ^ 80)
