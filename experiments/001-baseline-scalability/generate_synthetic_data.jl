"""
Generate synthetic datasets with known DAG structures for testing.

Creates 10 diverse datasets with different causal structures.
"""

using Random
using Statistics
using DelimitedFiles

Random.seed!(42)

function generate_linear_gaussian_data(structure::Dict{Int,Vector{Int}}, n_samples::Int=2000, noise_std::Float64=1.0)
    """Generate data from a linear Gaussian DAG."""
    n_vars = length(structure)
    data = zeros(n_samples, n_vars)

    # Topological sort
    visited = Set{Int}()
    topo_order = Int[]

    function dfs(node)
        if node in visited
            return
        end
        push!(visited, node)
        for parent in structure[node]
            dfs(parent)
        end
        push!(topo_order, node)
    end

    for node in 0:(n_vars-1)
        dfs(node)
    end

    # Generate data in topological order
    for node in topo_order
        parents = structure[node]
        if isempty(parents)
            # Root node
            data[:, node+1] = randn(n_samples) * noise_std
        else
            # Child node
            coefficients = rand(length(parents)) .* 1.5 .+ 0.5
            coefficients .*= rand([-1, 1], length(parents))

            for (i, p) in enumerate(parents)
                data[:, node+1] .+= coefficients[i] .* data[:, p+1]
            end
            data[:, node+1] .+= randn(n_samples) * noise_std
        end
    end

    return data
end

function save_dataset(name, structure, data, description)
    """Save synthetic dataset and metadata."""
    mkpath("synthetic_datasets")

    # Save data
    writedlm("synthetic_datasets/$(name).csv", data, ',')

    # Save metadata
    edges = sum(length(parents) for parents in values(structure))

    open("synthetic_datasets/$(name)_metadata.txt", "w") do f
        println(f, "Name: $name")
        println(f, "Description: $description")
        println(f, "Samples: $(size(data, 1))")
        println(f, "Variables: $(size(data, 2))")
        println(f, "Edges: $edges")
        println(f, "\nTrue Structure:")
        for (node, parents) in sort(collect(structure))
            if isempty(parents)
                println(f, "  $node: (no parents)")
            else
                println(f, "  $node: ← $(parents)")
            end
        end
    end

    println("Generated $name: $(size(data, 1)) samples × $(size(data, 2)) vars, $edges edges")
    println("  $description")
end

println("\n=== Generating Synthetic Datasets ===\n")

# Dataset 1: Empty graph
structure = Dict(i => Int[] for i in 0:9)
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("empty_graph", structure, data, "No edges - all variables independent")

# Dataset 2: Linear chain
structure = Dict(i => (i > 0 ? [i-1] : Int[]) for i in 0:9)
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("linear_chain", structure, data, "Chain: 0→1→2→3→4→5→6→7→8→9")

# Dataset 3: Star
structure = Dict(0 => Int[], [i => [0] for i in 1:9]...)
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("star", structure, data, "Star: 0 → {1,2,3,4,5,6,7,8,9}")

# Dataset 4: Colliders
structure = Dict(
    0 => Int[], 1 => Int[], 2 => [0, 1],
    3 => Int[], 4 => [3], 5 => [3],
    6 => Int[], 7 => [6], 8 => [6], 9 => Int[]
)
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("colliders", structure, data, "Multiple V-structures")

# Dataset 5: Fork
structure = Dict(
    0 => Int[],
    1 => [0], 2 => [0], 3 => [0],
    4 => [1, 2], 5 => [2, 3],
    6 => [4], 7 => [4], 8 => [5], 9 => [5]
)
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("fork", structure, data, "Common cause with convergence")

# Dataset 6: Dense random
Random.seed!(123)
structure = Dict{Int,Vector{Int}}()
for i in 0:9
    possible_parents = collect(0:(i-1))
    n_parents = min(4, length(possible_parents))
    if !isempty(possible_parents) && n_parents > 0
        structure[i] = sort(shuffle(possible_parents)[1:rand(0:n_parents)])
    else
        structure[i] = Int[]
    end
end
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("dense_random", structure, data, "Dense random DAG")

# Dataset 7: Sparse random
Random.seed!(456)
structure = Dict{Int,Vector{Int}}()
for i in 0:9
    possible_parents = collect(0:(i-1))
    n_parents = min(1, length(possible_parents))
    if !isempty(possible_parents) && rand() < 0.6
        structure[i] = [rand(possible_parents)]
    else
        structure[i] = Int[]
    end
end
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("sparse_random", structure, data, "Sparse random DAG")

# Dataset 8: Hierarchical
structure = Dict(
    0 => Int[], 1 => Int[], 2 => Int[],
    3 => [0, 1], 4 => [1, 2],
    5 => [3, 4], 6 => [3, 4],
    7 => [5], 8 => [6], 9 => [5, 6]
)
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("hierarchical", structure, data, "4-layer hierarchy")

# Dataset 9: Diamond
structure = Dict(
    0 => Int[],
    1 => [0], 2 => [0], 3 => [0],
    4 => [1, 2], 5 => [2, 3],
    6 => [4, 5],
    7 => [6], 8 => [6], 9 => [6]
)
data = generate_linear_gaussian_data(structure, 2000)
save_dataset("diamond", structure, data, "Diamond: convergent and divergent paths")

# Dataset 10: Mixed with constant
structure = Dict(
    0 => Int[],
    1 => [0], 2 => [1], 3 => [1],
    4 => [2, 3], 5 => Int[],
    6 => [5], 7 => [4, 6],
    8 => [7], 9 => [7]
)
data = generate_linear_gaussian_data(structure, 2000)
data[:, 1] .= 100.0  # Make first column constant
save_dataset("mixed_with_constant", structure, data, "Mixed with constant column")

println("\n=== Summary ===")
println("Generated 10 synthetic datasets in synthetic_datasets/")
println("Each has 2000 samples, 10 variables, known true structure")
