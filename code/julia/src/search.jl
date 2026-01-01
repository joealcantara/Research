"""
Structure search algorithms for causal discovery.

Functions for beam search and other search strategies to find high-scoring
DAG structures efficiently.
"""

using DataFrames
using Statistics
using DecisionTree

include("inference.jl")

"""
    custom_beam_search(df::DataFrame, target::Symbol, score_fn::Function;
                       k::Int=10, max_rounds::Int=3, verbose::Bool=false) -> Vector

Find high-scoring parent sets for a target variable using beam search.

# Arguments
- `df`: DataFrame
- `target`: Target variable (Symbol)
- `score_fn`: Function that takes a Vector{Symbol} of parents and returns a score
- `k`: Beam width (number of theories to keep each round)
- `max_rounds`: Maximum search rounds
- `verbose`: Print progress

# Returns
Vector of (parent_set, score) tuples sorted by score (best first)
"""
function custom_beam_search(df::DataFrame, target::Symbol, score_fn::Function;
                            k::Int=10, max_rounds::Int=3, verbose::Bool=false)
    predictors = [Symbol(c) for c in names(df) if Symbol(c) != target]

    # Initialize with single-variable theories + empty theory
    theories = [[v] for v in predictors]
    push!(theories, Symbol[])  # Empty theory

    scored = [(t, score_fn(t)) for t in theories]

    # Keep top k
    sort!(scored, by = x -> -x[2])
    current = scored[1:min(k, length(scored))]
    best_score = current[1][2]

    for round_num in 1:max_rounds
        candidates = []

        for (theory, old_score) in current
            # Keep original
            push!(candidates, (theory, old_score))

            # Try adding each predictor
            for v in predictors
                if !(v in theory)
                    new_theory = vcat(theory, [v])
                    new_score = score_fn(new_theory)
                    push!(candidates, (new_theory, new_score))
                end
            end
        end

        # Remove duplicates
        seen = Set()
        unique_candidates = []
        for (theory, score) in candidates
            key = Tuple(sort(theory))
            if !(key in seen)
                push!(seen, key)
                push!(unique_candidates, (theory, score))
            end
        end

        # Keep top k
        sort!(unique_candidates, by = x -> -x[2])
        current = unique_candidates[1:min(k, length(unique_candidates))]

        new_best = current[1][2]

        if verbose
            println("  Round $round_num: best score = $(round(new_best, digits=2))")
        end

        # Early stopping
        if new_best <= best_score + 0.1
            break
        end

        best_score = new_best
    end

    return current
end

"""
    score_theory_with_type(df::DataFrame, parent_vars::Vector{Symbol}, target::Symbol,
                           theory_type::Symbol, lambda_penalty::Float64) -> Float64

Score a theory (parent set → target) using specified theory type.

# Arguments
- `df`: DataFrame
- `parent_vars`: Parent variables
- `target`: Target variable
- `theory_type`: :linear or :tree
- `lambda_penalty`: Complexity penalty per edge

# Returns
Float64: Log-posterior score
"""
function score_theory_with_type(df::DataFrame, parent_vars::Vector{Symbol}, target::Symbol,
                                theory_type::Symbol, lambda_penalty::Float64)
    var_type = get_variable_type(df, target)

    if isempty(parent_vars)
        # No parents - marginal distribution (same for both theories)
        if var_type == :continuous
            μ = mean(df[!, target])
            σ = std(df[!, target]) + 1e-6
            log_lik = -0.5 * sum(log.(2 * π * σ^2) .+ ((df[!, target] .- μ) ./ σ).^2)
        else
            # Categorical marginal
            unique_vals = unique(df[!, target])
            total = nrow(df)
            num_values = length(unique_vals)
            log_lik = 0.0
            for val in unique_vals
                count = sum(df[!, target] .== val)
                prob = (count + 1) / (total + num_values)
                log_lik += count * log(prob)
            end
        end

        log_prior = 0.0  # No edges
        return log_lik + log_prior
    end

    # Has parents
    if theory_type == :linear
        if var_type == :continuous
            # Linear regression
            X = Matrix(df[!, parent_vars])
            y = df[!, target]
            X_with_intercept = hcat(ones(size(X, 1)), X)

            try
                coeffs = X_with_intercept \ y
                predictions = X_with_intercept * coeffs
                residuals = y - predictions
                residual_std = std(residuals) + 1e-6
                log_lik = -0.5 * sum(log.(2 * π * residual_std^2) .+ (residuals ./ residual_std).^2)
            catch
                log_lik = -Inf
            end
        else
            # Categorical with CPT
            parent_values = []
            for p in parent_vars
                p_type = get_variable_type(df, p)
                if p_type == :continuous
                    push!(parent_values, [0, 1])
                else
                    push!(parent_values, sort(unique(df[!, p])))
                end
            end

            all_combos = vec(collect(Iterators.product(parent_values...)))
            unique_vals = unique(df[!, target])
            num_values = length(unique_vals)

            log_lik = 0.0
            for combo in all_combos
                mask = trues(nrow(df))
                for (i, parent) in enumerate(parent_vars)
                    p_type = get_variable_type(df, parent)
                    if p_type == :continuous
                        median_val = median(df[!, parent])
                        if combo[i] == 0
                            mask .&= (df[!, parent] .<= median_val)
                        else
                            mask .&= (df[!, parent] .> median_val)
                        end
                    else
                        mask .&= (df[!, parent] .== combo[i])
                    end
                end

                subset = df[mask, :]
                if nrow(subset) == 0
                    continue
                end

                total = nrow(subset)
                for val in unique_vals
                    count = sum(subset[!, target] .== val)
                    prob = (count + 1) / (total + num_values)
                    log_lik += count * log(prob)
                end
            end
        end

    elseif theory_type == :tree
        # Decision tree
        X = Matrix(df[!, parent_vars])
        y = df[!, target]

        if var_type == :continuous
            tree = build_tree(y, X, maxdepth=2)
            predictions = apply_tree(tree, X)
            residuals = y - predictions
            residual_std = std(residuals) + 1e-6
            log_lik = -0.5 * sum(log.(2 * π * residual_std^2) .+ (residuals ./ residual_std).^2)
        else
            # Convert target to integers for classification
            y_int = Int.(y)
            tree = build_tree(y_int, X, maxdepth=2)

            log_lik = 0.0
            for i in 1:nrow(df)
                pred = apply_tree(tree, X[i:i, :])
                # Use predicted value as probability (simplified)
                # In practice, would use leaf probabilities
                log_lik += log(0.5 + 1e-10)  # Placeholder - simplified scoring
            end
        end
    end

    # Apply complexity penalty
    log_prior = -lambda_penalty * length(parent_vars)
    return log_lik + log_prior
end

"""
    beam_search_structure(df::DataFrame, variables::Vector{Symbol},
                         theory_assignment::Dict{Symbol, Symbol};
                         beam_size::Int=10, max_rounds::Int=5,
                         lambda_penalty::Float64=2.0) -> Tuple

Use beam search to find a good DAG structure given a theory assignment.

For each variable, find high-scoring parent sets using beam search.

# Arguments
- `df`: DataFrame
- `variables`: List of all variables
- `theory_assignment`: Dict mapping variables to theory types (:linear or :tree)
- `beam_size`: Beam width for search
- `max_rounds`: Maximum search rounds
- `lambda_penalty`: Complexity penalty per edge

# Returns
Tuple of (structure::Dict, total_score::Float64)
"""
function beam_search_structure(df::DataFrame, variables::Vector{Symbol},
                               theory_assignment::Dict{Symbol, Symbol};
                               beam_size::Int=10, max_rounds::Int=5,
                               lambda_penalty::Float64=2.0)
    structure = Dict{Symbol, Vector{Symbol}}()
    total_score = 0.0

    println("Running beam search for each variable (beam_size=$beam_size, max_rounds=$max_rounds)...")
    println()

    for var in variables
        theory_type = get(theory_assignment, var, :linear)

        # Custom scoring function with theory type
        function score_fn(parent_vars::Vector{Symbol})
            return score_theory_with_type(df, parent_vars, var, theory_type, lambda_penalty)
        end

        # Run beam search
        results = custom_beam_search(
            df, var, score_fn;
            k=beam_size, max_rounds=max_rounds,
            verbose=false
        )

        # Take best result
        if length(results) > 0
            best_parents, best_score = results[1]
            structure[var] = best_parents
            total_score += best_score

            if isempty(best_parents)
                println("  $var: (no parents) | score: $(round(best_score, digits=2)) | theory: $theory_type")
            else
                parents_str = join(string.(best_parents), ", ")
                println("  $var: ← $parents_str | score: $(round(best_score, digits=2)) | theory: $theory_type")
            end
        else
            structure[var] = Symbol[]
            println("  $var: (no parents - no results)")
        end
    end

    println()
    println("Total structure score: $(round(total_score, digits=2))")
    println()

    return structure, total_score
end
