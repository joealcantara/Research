"""
Bayesian inference for causal structure learning.

Functions for estimating parameters, computing likelihoods, and scoring
DAG structures against data using Bayesian inference.
"""

using DataFrames
using Statistics
using LinearAlgebra
using Distributions

"""
    get_variable_type(df::DataFrame, var::Symbol; max_categorical::Int=10) -> Symbol

Determine if a variable is :binary, :categorical, or :continuous.

# Arguments
- `df`: DataFrame
- `var`: Variable name (Symbol)
- `max_categorical`: Maximum unique values to consider categorical (default: 10)

# Returns
Symbol: :binary, :categorical, or :continuous
"""
function get_variable_type(df::DataFrame, var::Symbol; max_categorical::Int=10)
    unique_vals = unique(df[!, var])
    sort!(unique_vals)

    # Binary: exactly 2 values that are 0 and 1
    if length(unique_vals) == 2 && Set(unique_vals) == Set([0, 1])
        return :binary
    end

    # Categorical: discrete values, not too many unique values
    if length(unique_vals) <= max_categorical
        return :categorical
    end

    # Continuous: everything else
    return :continuous
end

"""
    estimate_parameters(df::DataFrame, structure::Dict{Symbol, Vector{Symbol}}) -> Dict

Estimate conditional probability tables/distributions from data.

Given a DAG structure and data, learn the conditional probabilities
for each variable given its parents. Handles binary, categorical, and continuous variables.

# Arguments
- `df`: DataFrame with variables (binary, categorical, or continuous)
- `structure`: Dict mapping variables to their parent lists
              e.g., Dict(:X1 => [], :X2 => [:X1], :X3 => [:X2])

# Returns
Dict of parameters for each variable:
- Binary variables: Dict(parent_combo => p) where p = P(var=1|parents)
- Categorical variables: Dict(parent_combo => Dict(value => prob))
- Continuous variables: Dict(() => Dict("type" => "gaussian", "mean" => μ, "std" => σ))
                       or Dict(() => Dict("type" => "linear", "coeffs" => β, "intercept" => α, "std" => σ))
"""
function estimate_parameters(df::DataFrame, structure::Dict{Symbol, Vector{Symbol}})
    params = Dict{Symbol, Any}()

    for (var, parents) in structure
        var_type = get_variable_type(df, var)

        if isempty(parents)
            # No parents: marginal distribution
            if var_type == :binary
                # Binary: P(var=1)
                count_1 = sum(df[!, var] .== 1)
                total = nrow(df)
                params[var] = Dict(() => (count_1 + 1) / (total + 2))

            elseif var_type == :categorical
                # Categorical: P(var=k) for each value k
                unique_vals = unique(df[!, var])
                total = nrow(df)
                num_values = length(unique_vals)
                prob_dist = Dict{Any, Float64}()
                for val in unique_vals
                    count = sum(df[!, var] .== val)
                    prob_dist[val] = (count + 1) / (total + num_values)
                end
                params[var] = Dict(() => prob_dist)

            else  # continuous
                # Continuous: Gaussian with empirical mean and std
                params[var] = Dict(() => Dict(
                    "type" => "gaussian",
                    "mean" => mean(df[!, var]),
                    "std" => std(df[!, var]) + 1e-6
                ))
            end

        else
            # Has parents
            if var_type == :continuous
                # Continuous variable: Linear regression
                X = Matrix(df[!, parents])
                y = df[!, var]

                # Simple linear regression: y = X @ coeffs + intercept
                # Add intercept column
                X_with_intercept = hcat(ones(size(X, 1)), X)

                # Solve: coeffs = (X^T X)^{-1} X^T y
                try
                    coeffs = X_with_intercept \ y
                    intercept = coeffs[1]
                    slopes = coeffs[2:end]

                    # Compute residuals
                    predictions = X_with_intercept * coeffs
                    residuals = y - predictions
                    residual_std = std(residuals) + 1e-6

                    params[var] = Dict(() => Dict(
                        "type" => "linear",
                        "intercept" => intercept,
                        "coeffs" => slopes,
                        "parents" => parents,
                        "std" => residual_std
                    ))
                catch e
                    # Fallback if regression fails
                    params[var] = Dict(() => Dict(
                        "type" => "gaussian",
                        "mean" => mean(df[!, var]),
                        "std" => std(df[!, var]) + 1e-6
                    ))
                end

            else
                # Binary or categorical with parents: conditional probability tables
                unique_vals = var_type == :categorical ? unique(df[!, var]) : nothing

                # Get parent values - discretize continuous parents to avoid explosion
                parent_values = []
                for p in parents
                    p_type = get_variable_type(df, p)
                    if p_type == :continuous
                        # Discretize continuous parent using median
                        push!(parent_values, [0, 1])  # Below/above median
                    else
                        push!(parent_values, sort(unique(df[!, p])))
                    end
                end

                # Generate all combinations of parent values
                all_combos = vec(collect(Iterators.product(parent_values...)))

                params[var] = Dict{Any, Any}()
                for combo in all_combos
                    # Filter data where parents match this combo
                    mask = trues(nrow(df))
                    for (i, parent) in enumerate(parents)
                        p_type = get_variable_type(df, parent)
                        if p_type == :continuous
                            # Discretize continuous: 0 = below median, 1 = above median
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

                    if var_type == :binary
                        # Binary: P(var=1 | parents)
                        count_1 = sum(subset[!, var] .== 1)
                        total = nrow(subset)
                        params[var][combo] = (count_1 + 1) / (total + 2)
                    else  # categorical
                        # Categorical: P(var=k | parents) for each value k
                        total = nrow(subset)
                        num_values = length(unique_vals)
                        prob_dist = Dict{Any, Float64}()
                        for val in unique_vals
                            count = sum(subset[!, var] .== val)
                            prob_dist[val] = (count + 1) / (total + num_values)
                        end
                        params[var][combo] = prob_dist
                    end
                end
            end
        end
    end

    return params
end

"""
    compute_log_likelihood(df::DataFrame, structure::Dict, params::Dict) -> Float64

Compute log probability of the data given the structure and parameters.

Calculates: log P(data | structure, params)

This is the sum over all rows of log P(row | structure, params), where
each row's probability is the product of P(each variable | its parents).

# Arguments
- `df`: DataFrame with variables (binary, categorical, or continuous)
- `structure`: DAG structure dict
- `params`: Parameters from estimate_parameters()

# Returns
Float64: Log-likelihood
"""
function compute_log_likelihood(df::DataFrame, structure::Dict{Symbol, Vector{Symbol}}, params::Dict)
    log_lik = 0.0

    for row in eachrow(df)
        for (var, parents) in structure
            var_type = get_variable_type(df, var)

            # Get parent values for this row
            if isempty(parents)
                parent_combo = ()
            else
                # Discretize continuous parents if needed (for categorical/binary variables)
                if var_type in [:binary, :categorical]
                    parent_combo = []
                    for p in parents
                        p_type = get_variable_type(df, p)
                        if p_type == :continuous
                            # Discretize: 0 if below median, 1 if above
                            median_val = median(df[!, p])
                            push!(parent_combo, row[p] <= median_val ? 0 : 1)
                        else
                            push!(parent_combo, row[p])
                        end
                    end
                    parent_combo = tuple(parent_combo...)
                else
                    parent_combo = tuple([row[p] for p in parents]...)
                end
            end

            # Get parameter for this variable
            param_value = get(params[var], parent_combo, get(params[var], (), nothing))
            if isnothing(param_value)
                # Fallback for continuous variables
                param_value = params[var][()]
            end

            # Determine how to compute likelihood based on parameter type
            if isa(param_value, Dict) && haskey(param_value, "type")
                # Continuous variable
                observed_val = row[var]

                if param_value["type"] == "gaussian"
                    # Gaussian: log N(x | mean, std^2)
                    μ = param_value["mean"]
                    σ = param_value["std"]
                    log_lik += -0.5 * log(2 * π * σ^2) - 0.5 * ((observed_val - μ) / σ)^2

                elseif param_value["type"] == "linear"
                    # Linear regression: predict, then Gaussian around prediction
                    parent_vals = [row[p] for p in param_value["parents"]]
                    prediction = param_value["intercept"] + dot(param_value["coeffs"], parent_vals)
                    σ = param_value["std"]
                    log_lik += -0.5 * log(2 * π * σ^2) - 0.5 * ((observed_val - prediction) / σ)^2
                end

            elseif isa(param_value, Dict)
                # Categorical: look up probability for observed value
                observed_val = row[var]
                prob = param_value[observed_val]
                log_lik += log(prob)

            else
                # Binary: param_value is P(var=1)
                if row[var] == 1
                    log_lik += log(param_value)
                else
                    log_lik += log(1 - param_value)
                end
            end
        end
    end

    return log_lik
end

"""
    compute_posteriors(df::DataFrame, structures::Dict, lambda_penalty::Float64=2.0) -> Dict

Compute posterior probability for each structure using Bayesian inference.

For each structure:
1. Estimate parameters (conditional probabilities)
2. Compute log-likelihood (how well it explains the data)
3. Apply complexity penalty (penalize more edges)
4. Compute posterior = likelihood × prior

# Arguments
- `df`: DataFrame with variables
- `structures`: Dict of {name => structure}
- `lambda_penalty`: Cost per edge in log-likelihood units (default: 2.0)

# Returns
Dict mapping structure names to results:
```
Dict(
    name => Dict(
        "log_likelihood" => float,
        "edges" => int,
        "log_prior" => float,
        "log_posterior" => float,
        "posterior" => float  # Normalized probability
    )
)
```
"""
function compute_posteriors(df::DataFrame, structures::Dict, lambda_penalty::Float64=2.0)
    results = Dict{Any, Dict{String, Any}}()

    for (name, structure) in structures
        params = estimate_parameters(df, structure)
        log_lik = compute_log_likelihood(df, structure, params)

        # Count edges for complexity penalty
        edges = sum(length(parents) for parents in values(structure))

        # Log posterior (unnormalized)
        log_prior = -lambda_penalty * edges
        log_posterior = log_lik + log_prior

        results[name] = Dict(
            "log_likelihood" => log_lik,
            "edges" => edges,
            "log_prior" => log_prior,
            "log_posterior" => log_posterior
        )
    end

    # Normalize to get probabilities
    log_posteriors = [r["log_posterior"] for r in values(results)]

    # Subtract max for numerical stability
    log_posteriors_shifted = log_posteriors .- maximum(log_posteriors)
    posteriors = exp.(log_posteriors_shifted)
    posteriors = posteriors ./ sum(posteriors)

    # Add normalized posteriors to results
    for (i, name) in enumerate(keys(results))
        results[name]["posterior"] = posteriors[i]
    end

    return results
end
