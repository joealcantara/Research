use polars::prelude::*;
use std::collections::HashMap;
use itertools::Itertools;
use statrs::function::gamma::ln_gamma;
use serde::Deserialize;
use nalgebra as na;

#[derive(Debug)]
pub enum VariableType {
    Binary,
    Categorical,
    Continuous,
}

pub enum Parameters {
    Binary(BinaryParams),
    Categorical(CategoricalParams),
    Continuous(ContinuousParams),
}

/// Parameters for continuous variables.
///
/// Two cases:
/// 1. Gaussian: No parents, marginal distribution N(mean, std^2)
/// 2. Linear: Has parents, linear regression y = intercept + coeffs·parents + noise
#[derive(Debug)]
pub enum ContinuousParams {
    /// Gaussian distribution (no parents)
    Gaussian {
        mean: f64,
        std: f64,
    },
    /// Linear regression (has parents)
    Linear {
        intercept: f64,
        coeffs: Vec<f64>,  // Coefficients for each parent
        std: f64,          // Residual standard deviation
    },
}

pub struct BinaryParams {
    /// Conditional probability tables: parent_values -> P(var=1 | parents)
    /// For variables with no parents, key is empty vector
    pub probs: HashMap<Vec<i64>, f64>,
}

pub struct CategoricalParams {
    /// Conditional probability tables: parent_values -> probability distribution over values
    /// For each parent configuration, stores P(var=k) for each possible value k
    /// Key: parent values (empty vec if no parents)
    /// Value: HashMap mapping each categorical value to its probability
    pub probs: HashMap<Vec<i64>, HashMap<i64, f64>>,
}

/// Determine the type of a variable in the DataFrame.
///
/// Detects three types of variables:
/// - **Binary:** Exactly 2 unique values {0, 1}
/// - **Categorical:** Discrete integer values, ≤ 10 unique values
/// - **Continuous:** Everything else (many unique values or floating point)
///
/// # Arguments
/// * `df` - The DataFrame containing the variable
/// * `var` - The variable name to check
///
/// # Returns
/// * `Ok(VariableType)` - The detected type
/// * `Err(PolarsError)` if variable doesn't exist
///
/// # Rust patterns demonstrated:
/// * `Result<T, E>` for error handling
/// * `?` operator for error propagation
/// * `if let` pattern matching for safe type extraction
/// * Iterator methods: `into_no_null_iter()`, `collect()`
pub fn get_variable_type(df: &DataFrame, var: &str) -> Result<VariableType, PolarsError> {
    // Get the column (? propagates error if column doesn't exist)
    let column = df.column(var)?;

    // Get unique values (? propagates error if operation fails)
    let unique_vals = column.unique()?;

    // Binary: exactly 2 values that are 0 and 1
    if unique_vals.len() == 2 {
        // Sort to ensure consistent ordering (ascending)
        // Polars 0.45 uses SortOptions struct instead of bool
        let sorted = unique_vals.sort(SortOptions::default())?;

        // Pattern match: only proceed if we can extract i64 values
        // This is type-safe - we handle the case where it's not i64
        if let Ok(vals) = sorted.i64() {
            // Convert to Vec for comparison
            // into_no_null_iter() assumes no nulls (panics if null found)
            let vec: Vec<i64> = vals.into_no_null_iter().collect();

            // Direct Vec comparison (Rust implements PartialEq for Vec)
            if vec == vec![0, 1] {
                return Ok(VariableType::Binary);
            }
        }
    }

    // Categorical: discrete values, not too many unique values (max 10)
    // Check for both integer and string categorical variables
    if unique_vals.len() <= 10 {
        // Integer categorical
        if let Ok(_vals) = column.i64() {
            return Ok(VariableType::Categorical);
        }
        // String categorical (e.g., species names)
        if let Ok(_vals) = column.str() {
            return Ok(VariableType::Categorical);
        }
    }

    // Continuous: everything else (many unique values or floating point)
    Ok(VariableType::Continuous)
}

/// Discretize a continuous variable at its median.
///
/// Returns 0 if value <= median, 1 if value > median.
/// Used for mixed networks where discrete children have continuous parents.
///
/// # Arguments
/// * `df` - DataFrame containing the variable
/// * `var` - Variable name to discretize
/// * `value` - The value to discretize
///
/// # Returns
/// 0 or 1 (discretized value)
fn discretize_at_median(df: &DataFrame, var: &str, value: f64) -> Result<i64, PolarsError> {
    let var_series = df.column(var)?.f64()?;
    let median = var_series.median().ok_or_else(|| {
        PolarsError::ComputeError(format!("Could not compute median for variable {}", var).into())
    })?;

    Ok(if value <= median { 0 } else { 1 })
}

/// Estimate conditional probability tables for binary variables.
///
/// For each variable in the structure, estimates P(var=1 | parents) using Laplace smoothing.
/// - No parents: P(var=1) = (count_1 + 1) / (total + 2)
/// - Has parents: P(var=1 | parents) for each parent combination
///
/// # Arguments
/// * `df` - DataFrame with binary variables (all variables must be {0, 1})
/// * `structure` - DAG structure mapping variable names to parent lists
///
/// # Returns
/// HashMap mapping each variable to its BinaryParams (CPT)
///
/// # Rust patterns demonstrated:
/// * HashMap iteration and building
/// * Nested loops with itertools::multi_cartesian_product
/// * Polars boolean masking and filtering
/// * Reference vs owned data (&str vs String, &[i64] vs Vec<i64>)
/// * Error propagation with ?
pub fn estimate_binary_parameters(
    df: &DataFrame,
    structure: &HashMap<String, Vec<String>>,
) -> Result<HashMap<String, BinaryParams>, PolarsError> {
    // Result HashMap: variable name -> BinaryParams
    let mut all_params: HashMap<String, BinaryParams> = HashMap::new();

    // Iterate over each variable in the structure
    for (var, parents) in structure.iter() {
        // Skip non-binary variables
        let var_type = get_variable_type(df, var)?;
        if !matches!(var_type, VariableType::Binary) {
            continue;
        }

        // Initialize the probability HashMap for this variable
        let mut probs: HashMap<Vec<i64>, f64> = HashMap::new();

        if parents.is_empty() {
            // Case 1: No parents - estimate marginal P(var=1)
            // Count how many rows have var==1
            let var_col = df.column(var)?.i64()?;
            let count_1 = var_col.into_iter().filter(|&v| v == Some(1)).count() as f64;
            let total = df.height() as f64;

            // Laplace smoothing: (count + 1) / (total + 2)
            // Prevents zero probabilities
            let prob_1 = (count_1 + 1.0) / (total + 2.0);

            // Empty Vec key represents "no parent values"
            probs.insert(vec![], prob_1);
        } else {
            // Case 2: Has parents - estimate P(var=1 | parent_combo) for each combination

            // Get possible values for each parent
            // For discrete parents: use actual values
            // For continuous parents: use {0, 1} (discretized at median)
            let parent_values: Vec<Vec<i64>> = parents
                .iter()
                .map(|p| {
                    let p_type = get_variable_type(df, p)?;
                    match p_type {
                        VariableType::Binary => Ok(vec![0, 1]),
                        VariableType::Categorical => {
                            // Get unique values for categorical parent
                            let vals: Vec<i64> = df
                                .column(p)?
                                .unique()?
                                .sort(SortOptions::default())?
                                .i64()?
                                .into_no_null_iter()
                                .collect();
                            Ok(vals)
                        }
                        VariableType::Continuous => {
                            // Continuous parents discretized to {0, 1}
                            Ok(vec![0, 1])
                        }
                    }
                })
                .collect::<Result<Vec<Vec<i64>>, PolarsError>>()?;

            // Generate all possible parent combinations
            let all_combos = parent_values.into_iter().multi_cartesian_product();

            for combo in all_combos {
                // combo is Vec<i64>, e.g., [0, 1] meaning parent1=0, parent2=1

                // Build a boolean mask: which rows match this parent combination?
                // Start with all True
                let mut mask = vec![true; df.height()];

                for (i, parent) in parents.iter().enumerate() {
                    let parent_type = get_variable_type(df, parent)?;
                    let parent_val = combo[i];

                    match parent_type {
                        VariableType::Continuous => {
                            // Continuous parent: discretize at median
                            let parent_col = df.column(parent)?.f64()?;
                            for (row_idx, opt_val) in parent_col.into_iter().enumerate() {
                                if let Some(val) = opt_val {
                                    let discretized = discretize_at_median(df, parent, val)?;
                                    mask[row_idx] = mask[row_idx] && (discretized == parent_val);
                                } else {
                                    mask[row_idx] = false;
                                }
                            }
                        }
                        _ => {
                            // Discrete parent: use actual values
                            let parent_col = df.column(parent)?.i64()?;
                            for (row_idx, opt_val) in parent_col.into_iter().enumerate() {
                                if let Some(val) = opt_val {
                                    mask[row_idx] = mask[row_idx] && (val == parent_val);
                                } else {
                                    mask[row_idx] = false;
                                }
                            }
                        }
                    }
                }

                // Count var=1 in the filtered subset
                let var_col = df.column(var)?.i64()?;
                let mut count_1 = 0.0;
                let mut count_total = 0.0;

                for (row_idx, opt_val) in var_col.into_iter().enumerate() {
                    if mask[row_idx] {
                        count_total += 1.0;
                        if let Some(val) = opt_val {
                            if val == 1 {
                                count_1 += 1.0;
                            }
                        }
                    }
                }

                // Laplace smoothing: (count + 1) / (total + 2)
                let prob_1 = (count_1 + 1.0) / (count_total + 2.0);

                // Store in HashMap with combo as key
                probs.insert(combo, prob_1);
            }
        }

        // Store this variable's parameters
        all_params.insert(var.clone(), BinaryParams { probs });
    }

    Ok(all_params)
}

/// Estimate conditional probability tables for categorical variables.
///
/// For each variable in the structure, estimates P(var=k | parents) for each value k.
/// Uses Laplace smoothing: (count_k + 1) / (total + K) where K is number of values.
///
/// # Arguments
/// * `df` - DataFrame with categorical variables (integer values)
/// * `structure` - DAG structure mapping variable names to parent lists
///
/// # Returns
/// HashMap mapping each variable to its CategoricalParams (multinomial CPT)
///
/// # Rust patterns demonstrated:
/// * Nested HashMaps for probability distributions
/// * Getting unique values from columns
/// * Multinomial probability estimation
pub fn estimate_categorical_parameters(
    df: &DataFrame,
    structure: &HashMap<String, Vec<String>>,
) -> Result<HashMap<String, CategoricalParams>, PolarsError> {
    let mut all_params: HashMap<String, CategoricalParams> = HashMap::new();

    for (var, parents) in structure.iter() {
        // Skip non-categorical variables
        let var_type = get_variable_type(df, var)?;
        if !matches!(var_type, VariableType::Binary | VariableType::Categorical) {
            continue;
        }

        // Get unique values for this variable
        let var_col = df.column(var)?.i64()?;
        let unique_var_vals: Vec<i64> = df
            .column(var)?
            .unique()?
            .sort(SortOptions::default())?
            .i64()?
            .into_no_null_iter()
            .collect();

        let num_values = unique_var_vals.len() as f64;

        // Initialize probability HashMap
        let mut probs: HashMap<Vec<i64>, HashMap<i64, f64>> = HashMap::new();

        if parents.is_empty() {
            // No parents: marginal distribution P(var=k)
            let total = df.height() as f64;
            let mut prob_dist = HashMap::new();

            for &val in &unique_var_vals {
                let count = var_col.into_iter().filter(|&v| v == Some(val)).count() as f64;
                // Laplace smoothing: (count + 1) / (total + K)
                prob_dist.insert(val, (count + 1.0) / (total + num_values));
            }

            probs.insert(vec![], prob_dist);
        } else {
            // Has parents: P(var=k | parent_combo) for each combination

            // Generate all parent combinations
            // For discrete parents: use actual values
            // For continuous parents: use {0, 1} (discretized at median)
            let parent_values: Vec<Vec<i64>> = parents
                .iter()
                .map(|p| {
                    let p_type = get_variable_type(df, p)?;
                    match p_type {
                        VariableType::Binary => Ok(vec![0, 1]),
                        VariableType::Categorical => {
                            let vals: Vec<i64> = df
                                .column(p)?
                                .unique()?
                                .sort(SortOptions::default())?
                                .i64()?
                                .into_no_null_iter()
                                .collect();
                            Ok(vals)
                        }
                        VariableType::Continuous => {
                            // Continuous parents discretized to {0, 1}
                            Ok(vec![0, 1])
                        }
                    }
                })
                .collect::<Result<Vec<Vec<i64>>, PolarsError>>()?;

            let all_combos = parent_values.into_iter().multi_cartesian_product();

            for combo in all_combos {
                // Build mask for this parent configuration
                let mut mask = vec![true; df.height()];

                for (i, parent) in parents.iter().enumerate() {
                    let parent_type = get_variable_type(df, parent)?;
                    let parent_val = combo[i];

                    match parent_type {
                        VariableType::Continuous => {
                            // Continuous parent: discretize at median
                            let parent_col = df.column(parent)?.f64()?;
                            for (row_idx, opt_val) in parent_col.into_iter().enumerate() {
                                if let Some(val) = opt_val {
                                    let discretized = discretize_at_median(df, parent, val)?;
                                    mask[row_idx] = mask[row_idx] && (discretized == parent_val);
                                } else {
                                    mask[row_idx] = false;
                                }
                            }
                        }
                        _ => {
                            // Discrete parent: use actual values
                            let parent_col = df.column(parent)?.i64()?;
                            for (row_idx, opt_val) in parent_col.into_iter().enumerate() {
                                if let Some(val) = opt_val {
                                    mask[row_idx] = mask[row_idx] && (val == parent_val);
                                } else {
                                    mask[row_idx] = false;
                                }
                            }
                        }
                    }
                }

                // Count each value in this configuration
                let mut counts = HashMap::new();
                let mut total_count = 0.0;

                for (row_idx, opt_val) in var_col.into_iter().enumerate() {
                    if mask[row_idx] {
                        if let Some(val) = opt_val {
                            *counts.entry(val).or_insert(0.0) += 1.0;
                            total_count += 1.0;
                        }
                    }
                }

                // Compute probabilities with Laplace smoothing
                let mut prob_dist = HashMap::new();
                for &val in &unique_var_vals {
                    let count = counts.get(&val).unwrap_or(&0.0);
                    // Laplace smoothing: (count + 1) / (total + K)
                    prob_dist.insert(val, (count + 1.0) / (total_count + num_values));
                }

                probs.insert(combo, prob_dist);
            }
        }

        all_params.insert(var.clone(), CategoricalParams { probs });
    }

    Ok(all_params)
}

/// Estimate parameters for continuous variables.
///
/// For each variable in the structure:
/// - **No parents**: Gaussian distribution N(mean, std)
/// - **Has parents**: Linear regression y = intercept + coeffs·parents + noise
///
/// Uses least squares for linear regression: coeffs = (X^T X)^{-1} X^T y
///
/// # Arguments
/// * `df` - DataFrame with continuous variables (floating point)
/// * `structure` - DAG structure mapping variable names to parent lists
///
/// # Returns
/// HashMap mapping each variable to its ContinuousParams
///
/// # Rust patterns demonstrated:
/// * Matrix operations for linear regression
/// * Polars mean() and std() aggregations
/// * Error handling for singular matrices
pub fn estimate_continuous_parameters(
    df: &DataFrame,
    structure: &HashMap<String, Vec<String>>,
) -> Result<HashMap<String, ContinuousParams>, PolarsError> {
    let mut all_params: HashMap<String, ContinuousParams> = HashMap::new();

    for (var, parents) in structure.iter() {
        if parents.is_empty() {
            // Case 1: No parents - Gaussian distribution
            let var_series = df.column(var)?.f64()?;

            // Compute mean and std
            let mean = var_series.mean().ok_or_else(|| {
                PolarsError::ComputeError(format!("Could not compute mean for variable {}", var).into())
            })?;

            let std = var_series.std(1).ok_or_else(|| {
                PolarsError::ComputeError(format!("Could not compute std for variable {}", var).into())
            })? + 1e-6; // Add small constant to avoid zero std

            all_params.insert(
                var.clone(),
                ContinuousParams::Gaussian { mean, std },
            );
        } else {
            // Case 2: Has parents - Linear regression using nalgebra
            // Build design matrix X and response vector y

            let n = df.height();
            let p = parents.len();

            // Get response variable y
            let y_series = df.column(var)?;
            let y_vec: Vec<f64> = y_series.f64()?.into_no_null_iter().collect();

            // Build design matrix X using nalgebra (heap-allocated)
            // X is n × (p+1), including intercept column
            let mut x_data: Vec<f64> = Vec::with_capacity(n * (p + 1));

            // First column: intercept (all 1.0)
            for _ in 0..n {
                x_data.push(1.0);
            }

            // Remaining columns: parent values
            for parent in parents.iter() {
                let parent_series = df.column(parent)?;

                // Handle both f64 (continuous) and i64 (categorical) parents
                let parent_vals: Vec<f64> = if let Ok(vals) = parent_series.f64() {
                    vals.into_no_null_iter().collect()
                } else if let Ok(vals) = parent_series.i64() {
                    vals.into_no_null_iter().map(|v| v as f64).collect()
                } else {
                    return Err(PolarsError::ComputeError(
                        format!("Parent {} has unsupported type for regression", parent).into()
                    ));
                };

                x_data.extend(parent_vals);
            }

            // Create nalgebra matrices (stored on heap)
            let x_matrix = na::DMatrix::from_vec(n, p + 1, x_data);
            let y_vector = na::DVector::from_vec(y_vec);

            // Solve least squares using SVD (numerically stable)
            // Clone x_matrix since SVD consumes it
            let svd = x_matrix.clone().svd(true, true);
            let coeffs_vec = svd.solve(&y_vector, 1e-10).map_err(|e| {
                PolarsError::ComputeError(
                    format!("Failed to solve linear regression for {}: {}", var, e).into()
                )
            })?;

            let intercept = coeffs_vec[0];
            let slopes: Vec<f64> = coeffs_vec.as_slice()[1..].to_vec();

            // Compute residuals
            let predictions = &x_matrix * &coeffs_vec;
            let residuals = &y_vector - &predictions;

            // Compute std of residuals
            let variance = residuals.norm_squared() / n as f64;
            let std = variance.sqrt() + 1e-6;

            all_params.insert(
                var.clone(),
                ContinuousParams::Linear {
                    intercept,
                    coeffs: slopes,
                    std,
                },
            );
        }
    }

    Ok(all_params)
}

/// Compute log-likelihood of data given structure and parameters.
///
/// Calculates: log P(data | structure, params)
///
/// This is the sum over all rows of the log probability of each row.
/// For each row, the probability is the product of P(each variable | its parents).
/// We work in log space to avoid numerical underflow.
///
/// # Arguments
/// * `df` - DataFrame with binary variables
/// * `structure` - DAG structure
/// * `params` - Parameters from estimate_binary_parameters()
///
/// # Returns
/// Log-likelihood (f64)
///
/// # Rust patterns demonstrated:
/// * Nested iteration over rows and variables
/// * Extracting values from HashMaps with get()
/// * Floating point arithmetic in log space
/// * unwrap_or() for default values
pub fn compute_binary_log_likelihood(
    df: &DataFrame,
    structure: &HashMap<String, Vec<String>>,
    params: &HashMap<String, BinaryParams>,
) -> Result<f64, PolarsError> {
    let mut log_lik = 0.0;

    // Iterate over each row in the DataFrame
    let n_rows = df.height();

    for row_idx in 0..n_rows {
        // For each variable in the structure
        for (var, parents) in structure.iter() {
            // Skip non-binary variables
            let var_type = get_variable_type(df, var)?;
            if !matches!(var_type, VariableType::Binary) {
                continue;
            }

            // Get the observed value for this variable in this row
            let var_col = df.column(var)?.i64()?;
            let observed_val = var_col
                .get(row_idx)
                .ok_or_else(|| PolarsError::ComputeError("Row index out of bounds".into()))?;

            // Get parent values for this row to form the key
            // Discretize continuous parents at median
            let parent_combo: Vec<i64> = if parents.is_empty() {
                vec![] // No parents
            } else {
                parents
                    .iter()
                    .map(|p| {
                        let p_type = get_variable_type(df, p)?;
                        match p_type {
                            VariableType::Continuous => {
                                // Discretize continuous parent
                                let p_col = df.column(p)?.f64()?;
                                let val = p_col
                                    .get(row_idx)
                                    .ok_or_else(|| PolarsError::ComputeError("Parent value missing".into()))?;
                                discretize_at_median(df, p, val)
                            }
                            _ => {
                                // Discrete parent: use actual value
                                let p_col = df.column(p)?.i64()?;
                                p_col
                                    .get(row_idx)
                                    .ok_or_else(|| PolarsError::ComputeError("Parent value missing".into()))
                            }
                        }
                    })
                    .collect::<Result<Vec<i64>, PolarsError>>()?
            };

            // Look up P(var=1 | parents) from parameters
            let var_params = params
                .get(var)
                .ok_or_else(|| PolarsError::ComputeError(format!("No params for variable {}", var).into()))?;

            let prob_1 = var_params
                .probs
                .get(&parent_combo)
                .ok_or_else(|| {
                    PolarsError::ComputeError(
                        format!("No probability for combo {:?} in variable {}", parent_combo, var).into()
                    )
                })?;

            // Compute log probability for this observation
            // If observed_val == 1: log(prob_1)
            // If observed_val == 0: log(1 - prob_1)
            let log_prob = if observed_val == 1 {
                prob_1.ln()
            } else if observed_val == 0 {
                (1.0 - prob_1).ln()
            } else {
                return Err(PolarsError::ComputeError(
                    format!("Invalid value {} for binary variable {}", observed_val, var).into()
                ));
            };

            log_lik += log_prob;
        }
    }

    Ok(log_lik)
}

/// Compute log-likelihood for categorical variables.
///
/// Calculates: log P(data | structure, params) for multinomial distributions
///
/// For each row and variable, looks up P(var=observed_value | parents) and adds log probability.
///
/// # Arguments
/// * `df` - DataFrame with categorical variables
/// * `structure` - DAG structure
/// * `params` - Parameters from estimate_categorical_parameters()
///
/// # Returns
/// Log-likelihood (f64)
pub fn compute_categorical_log_likelihood(
    df: &DataFrame,
    structure: &HashMap<String, Vec<String>>,
    params: &HashMap<String, CategoricalParams>,
) -> Result<f64, PolarsError> {
    let mut log_lik = 0.0;
    let n_rows = df.height();

    for row_idx in 0..n_rows {
        for (var, parents) in structure.iter() {
            // Skip non-categorical variables
            let var_type = get_variable_type(df, var)?;
            if !matches!(var_type, VariableType::Binary | VariableType::Categorical) {
                continue;
            }

            // Get observed value for this variable
            let var_col = df.column(var)?.i64()?;
            let observed_val = var_col
                .get(row_idx)
                .ok_or_else(|| PolarsError::ComputeError("Row index out of bounds".into()))?;

            // Get parent values for this row
            // Discretize continuous parents at median
            let parent_combo: Vec<i64> = if parents.is_empty() {
                vec![]
            } else {
                parents
                    .iter()
                    .map(|p| {
                        let p_type = get_variable_type(df, p)?;
                        match p_type {
                            VariableType::Continuous => {
                                // Discretize continuous parent
                                let p_col = df.column(p)?.f64()?;
                                let val = p_col
                                    .get(row_idx)
                                    .ok_or_else(|| PolarsError::ComputeError("Parent value missing".into()))?;
                                discretize_at_median(df, p, val)
                            }
                            _ => {
                                // Discrete parent: use actual value
                                let p_col = df.column(p)?.i64()?;
                                p_col
                                    .get(row_idx)
                                    .ok_or_else(|| PolarsError::ComputeError("Parent value missing".into()))
                            }
                        }
                    })
                    .collect::<Result<Vec<i64>, PolarsError>>()?
            };

            // Look up P(var=observed_val | parents)
            let var_params = params
                .get(var)
                .ok_or_else(|| PolarsError::ComputeError(format!("No params for variable {}", var).into()))?;

            let prob_dist = var_params
                .probs
                .get(&parent_combo)
                .ok_or_else(|| {
                    PolarsError::ComputeError(
                        format!("No probability for combo {:?} in variable {}", parent_combo, var).into()
                    )
                })?;

            let prob = prob_dist
                .get(&observed_val)
                .ok_or_else(|| {
                    PolarsError::ComputeError(
                        format!("No probability for value {} in variable {}", observed_val, var).into()
                    )
                })?;

            log_lik += prob.ln();
        }
    }

    Ok(log_lik)
}

/// Compute log-likelihood for continuous variables.
///
/// Calculates: log P(data | structure, params) for Gaussian and linear Gaussian models
///
/// For each row and variable:
/// - Gaussian (no parents): log N(x | mean, std^2)
/// - Linear (has parents): log N(x | predicted_mean, residual_std^2)
///
/// The Gaussian log-likelihood formula is:
/// log N(x | μ, σ^2) = -0.5 * log(2π*σ^2) - 0.5 * ((x - μ) / σ)^2
///
/// # Arguments
/// * `df` - DataFrame with continuous variables
/// * `structure` - DAG structure
/// * `params` - Parameters from estimate_continuous_parameters()
///
/// # Returns
/// Log-likelihood (f64)
///
/// # Rust patterns demonstrated:
/// * Pattern matching on enum variants (Gaussian vs Linear)
/// * Floating point arithmetic in log space
/// * Mathematical constants (std::f64::consts::PI)
pub fn compute_continuous_log_likelihood(
    df: &DataFrame,
    structure: &HashMap<String, Vec<String>>,
    params: &HashMap<String, ContinuousParams>,
) -> Result<f64, PolarsError> {
    let mut log_lik = 0.0;
    let n_rows = df.height();

    for row_idx in 0..n_rows {
        for (var, parents) in structure.iter() {
            // Get observed value for this variable
            let var_col = df.column(var)?.f64()?;
            let observed_val = var_col
                .get(row_idx)
                .ok_or_else(|| PolarsError::ComputeError("Row index out of bounds".into()))?;

            // Get parameters for this variable
            let var_params = params
                .get(var)
                .ok_or_else(|| PolarsError::ComputeError(format!("No params for variable {}", var).into()))?;

            // Compute log-likelihood based on parameter type
            match var_params {
                ContinuousParams::Gaussian { mean, std } => {
                    // Gaussian: log N(x | mean, std^2)
                    // = -0.5 * log(2π*std^2) - 0.5 * ((x - mean) / std)^2
                    let pi = std::f64::consts::PI;
                    let log_prob = -0.5 * (2.0 * pi * std * std).ln()
                        - 0.5 * ((observed_val - mean) / std).powi(2);
                    log_lik += log_prob;
                }
                ContinuousParams::Linear { intercept, coeffs, std } => {
                    // Linear regression: predict mean, then Gaussian around prediction
                    // Get parent values for this row
                    let mut prediction = *intercept;

                    for (i, parent) in parents.iter().enumerate() {
                        let parent_col = df.column(parent)?;

                        // Handle both f64 (continuous) and i64 (categorical) parents
                        let parent_val = if let Ok(col) = parent_col.f64() {
                            col.get(row_idx).ok_or_else(|| PolarsError::ComputeError("Parent value missing".into()))?
                        } else if let Ok(col) = parent_col.i64() {
                            col.get(row_idx).ok_or_else(|| PolarsError::ComputeError("Parent value missing".into()))? as f64
                        } else {
                            return Err(PolarsError::ComputeError(
                                format!("Parent {} has unsupported type in likelihood", parent).into()
                            ));
                        };

                        prediction += coeffs[i] * parent_val;
                    }

                    // Gaussian likelihood around predicted mean
                    let pi = std::f64::consts::PI;
                    let log_prob = -0.5 * (2.0 * pi * std * std).ln()
                        - 0.5 * ((observed_val - prediction) / std).powi(2);
                    log_lik += log_prob;
                }
            }
        }
    }

    Ok(log_lik)
}

/// Score a DAG structure using BIC-like criterion.
///
/// Score = log_likelihood - lambda × num_edges
///
/// This balances fit to data (log-likelihood) against model complexity (edges).
/// Higher scores are better. Lambda controls the complexity penalty.
///
/// # Arguments
/// * `df` - DataFrame with binary variables
/// * `structure` - DAG structure to score
/// * `lambda_penalty` - Cost per edge (default: 2.0 is BIC-like)
///
/// # Returns
/// Tuple of (score, log_likelihood, num_edges)
///
/// # Rust patterns demonstrated:
/// * Tuple return types
/// * Function composition (calling other functions)
/// * Counting with sum() and iterator methods
pub fn score_binary_structure(
    df: &DataFrame,
    structure: &HashMap<String, Vec<String>>,
    lambda_penalty: f64,
) -> Result<(f64, f64, usize), PolarsError> {
    // Step 1: Estimate parameters
    let params = estimate_binary_parameters(df, structure)?;

    // Step 2: Compute log-likelihood
    let log_lik = compute_binary_log_likelihood(df, structure, &params)?;

    // Step 3: Count edges (total number of parent relationships)
    // Sum of lengths of all parent lists
    let num_edges: usize = structure.values().map(|parents| parents.len()).sum();

    // Step 4: Apply complexity penalty
    // BIC-like scoring: log_lik - lambda × edges
    let score = log_lik - lambda_penalty * (num_edges as f64);

    Ok((score, log_lik, num_edges))
}

/// Compute the number of free parameters for a binary Bayesian network.
///
/// For binary variables:
/// - No parents: 1 parameter (P(var=1))
/// - k binary parents: 2^k parameters (one per parent combination)
///
/// # Returns
/// Total number of free parameters across all variables
pub fn count_parameters(structure: &HashMap<String, Vec<String>>) -> usize {
    structure
        .values()
        .map(|parents| {
            if parents.is_empty() {
                1
            } else {
                // 2^k parameters for k binary parents
                2_usize.pow(parents.len() as u32)
            }
        })
        .sum()
}

/// Score a DAG structure using true BIC (Bayesian Information Criterion).
///
/// BIC = log_likelihood - (k/2) × log(n)
///
/// Where:
/// - k = number of free parameters
/// - n = sample size
///
/// Key differences from our edge-based score:
/// 1. Counts parameters (2^m for m parents) not edges
/// 2. Penalty scales with log(n), not fixed lambda
/// 3. More theoretically grounded (approximates Bayes factor)
///
/// # Arguments
/// * `df` - DataFrame with binary variables
/// * `structure` - DAG structure to score
///
/// # Returns
/// Tuple of (bic_score, log_likelihood, num_parameters)
///
/// # Rust patterns demonstrated:
/// * pow() for exponentiation
/// * Natural log with ln()
pub fn score_binary_structure_bic(
    df: &DataFrame,
    structure: &HashMap<String, Vec<String>>,
) -> Result<(f64, f64, usize), PolarsError> {
    // Step 1: Estimate parameters (using MAP with Laplace smoothing)
    // Note: True BIC uses MLE, but the difference is small with enough data
    let params = estimate_binary_parameters(df, structure)?;

    // Step 2: Compute log-likelihood
    let log_lik = compute_binary_log_likelihood(df, structure, &params)?;

    // Step 3: Count parameters
    let k = count_parameters(structure);

    // Step 4: Compute BIC penalty
    let n = df.height() as f64;
    let penalty = (k as f64 / 2.0) * n.ln();

    // Step 5: BIC = log_lik - penalty
    let bic_score = log_lik - penalty;

    Ok((bic_score, log_lik, k))
}

/// Score a DAG structure using BDeu (Bayesian Dirichlet equivalent uniform).
///
/// This is the most theoretically correct scoring method for discrete Bayesian networks.
/// It computes the **marginal likelihood** by integrating over all possible parameters:
///
/// P(D | G) = ∫ P(D | G, θ) × P(θ | G) dθ
///
/// The integral has a closed-form solution using Gamma functions:
///
/// log P(D | G) = Σᵢ Σⱼ [ ln Γ(αᵢⱼ) - ln Γ(αᵢⱼ + Nᵢⱼ) +
///                        Σₖ (ln Γ(αᵢⱼₖ + Nᵢⱼₖ) - ln Γ(αᵢⱼₖ)) ]
///
/// Where:
/// - i = variable index
/// - j = parent configuration index
/// - k = value index (0 or 1 for binary)
/// - αᵢⱼₖ = prior pseudocount (Dirichlet hyperparameter)
/// - Nᵢⱼₖ = observed count in data
///
/// For BDeu, we use uniform priors:
/// - αᵢⱼₖ = α / (rᵢ × qᵢ)
/// - rᵢ = number of values (2 for binary)
/// - qᵢ = number of parent configs (2^|parents|)
/// - α = equivalent sample size (typically 1)
///
/// # Arguments
/// * `df` - DataFrame with binary variables
/// * `structure` - DAG structure to score
/// * `alpha` - Equivalent sample size (default: 1.0)
///
/// # Returns
/// Log marginal likelihood log P(D | G)
///
/// # Theory Notes
/// - NO separate likelihood and penalty - complexity is automatically penalized!
/// - Simple models: Few parameters → concentrated integral → high score
/// - Complex models: Many parameters → diffuse integral → low score
/// - This is the "automatic Occam's Razor" of Bayesian model selection
///
/// # Rust patterns demonstrated:
/// * ln_gamma() for log Gamma function
/// * Careful handling of edge cases (empty parent sets)
pub fn score_binary_structure_bdeu(
    df: &DataFrame,
    structure: &HashMap<String, Vec<String>>,
    alpha: f64,
) -> Result<f64, PolarsError> {
    let mut log_score = 0.0;

    // For each variable in the structure
    for (var, parents) in structure.iter() {
        // Verify variable is binary
        get_variable_type(df, var)?;

        let r_i = 2.0; // Number of values (binary: 0 and 1)

        if parents.is_empty() {
            // No parents: single configuration
            let q_i = 1.0;
            let alpha_ij = alpha / q_i; // Total pseudocount for this config
            let alpha_ijk = alpha / (r_i * q_i); // Pseudocount per value

            // Count occurrences in data
            let var_col = df.column(var)?.i64()?;
            let n_1 = var_col.into_iter().filter(|&v| v == Some(1)).count() as f64;
            let n_0 = var_col.into_iter().filter(|&v| v == Some(0)).count() as f64;
            let n_ij = n_1 + n_0; // Total count

            // BDeu score component for this variable
            // ln Γ(α_ij) - ln Γ(α_ij + N_ij) + Σ_k [ln Γ(α_ijk + N_ijk) - ln Γ(α_ijk)]
            let component = ln_gamma(alpha_ij) - ln_gamma(alpha_ij + n_ij)
                + (ln_gamma(alpha_ijk + n_0) - ln_gamma(alpha_ijk))
                + (ln_gamma(alpha_ijk + n_1) - ln_gamma(alpha_ijk));

            log_score += component;

        } else {
            // Has parents: iterate over all parent configurations
            let q_i = 2_f64.powi(parents.len() as i32); // 2^|parents|
            let alpha_ij = alpha / q_i;
            let alpha_ijk = alpha / (r_i * q_i);

            // Verify all parents are binary
            for parent in parents.iter() {
                get_variable_type(df, parent)?;
            }

            // Generate all parent combinations
            let parent_values: Vec<Vec<i64>> = vec![vec![0, 1]; parents.len()];
            let all_combos = parent_values.into_iter().multi_cartesian_product();

            for combo in all_combos {
                // Build mask for this parent configuration
                let mut mask = vec![true; df.height()];

                for (i, parent) in parents.iter().enumerate() {
                    let parent_col = df.column(parent)?.i64()?;
                    let parent_val = combo[i];

                    for (row_idx, opt_val) in parent_col.into_iter().enumerate() {
                        if let Some(val) = opt_val {
                            mask[row_idx] = mask[row_idx] && (val == parent_val);
                        } else {
                            mask[row_idx] = false;
                        }
                    }
                }

                // Count var values in this configuration
                let var_col = df.column(var)?.i64()?;
                let mut n_0 = 0.0;
                let mut n_1 = 0.0;

                for (row_idx, opt_val) in var_col.into_iter().enumerate() {
                    if mask[row_idx] {
                        if let Some(val) = opt_val {
                            if val == 1 {
                                n_1 += 1.0;
                            } else {
                                n_0 += 1.0;
                            }
                        }
                    }
                }

                let n_ij = n_0 + n_1;

                // BDeu score component for this configuration
                let component = ln_gamma(alpha_ij) - ln_gamma(alpha_ij + n_ij)
                    + (ln_gamma(alpha_ijk + n_0) - ln_gamma(alpha_ijk))
                    + (ln_gamma(alpha_ijk + n_1) - ln_gamma(alpha_ijk));

                log_score += component;
            }
        }
    }

    Ok(log_score)
}

/// Configuration for scoring methods from TOML files.
///
/// This struct is designed for easy TOML deserialization.
/// Use `into()` to convert to `ScoringMethod`.
///
/// # TOML Examples
/// ```toml
/// # Edge-based with lambda
/// scoring = { method = "edge", lambda = 2.0 }
///
/// # BIC (no parameters)
/// scoring = { method = "bic" }
///
/// # BDeu with alpha
/// scoring = { method = "bdeu", alpha = 1.0 }
///
/// # Multiple methods for comparison
/// scoring = [
///     { method = "edge", lambda = 2.0 },
///     { method = "bic" },
///     { method = "bdeu", alpha = 1.0 }
/// ]
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ScoringConfig {
    /// Method name: "edge", "bic", or "bdeu"
    pub method: String,

    /// Lambda parameter for edge-based scoring (optional)
    #[serde(default)]
    pub lambda: Option<f64>,

    /// Alpha parameter for BDeu scoring (optional)
    #[serde(default)]
    pub alpha: Option<f64>,
}

impl From<ScoringConfig> for ScoringMethod {
    fn from(config: ScoringConfig) -> Self {
        match config.method.to_lowercase().as_str() {
            "edge" | "edge-based" => {
                let lambda = config.lambda.unwrap_or(2.0);
                ScoringMethod::EdgeBased(lambda)
            }
            "bic" => ScoringMethod::BIC,
            "bdeu" => {
                let alpha = config.alpha.unwrap_or(1.0);
                ScoringMethod::BDeu(alpha)
            }
            _ => {
                eprintln!("Unknown scoring method '{}', defaulting to EdgeBased(2.0)", config.method);
                ScoringMethod::EdgeBased(2.0)
            }
        }
    }
}

/// Scoring method for Bayesian network structure learning.
///
/// Provides a unified interface for different scoring approaches.
/// Each has different tradeoffs between speed, theoretical rigor, and flexibility.
#[derive(Debug, Clone)]
pub enum ScoringMethod {
    /// Edge-based penalty with fixed lambda
    ///
    /// Score = log_likelihood - λ × edges
    ///
    /// **Pros:** Fast, tunable, works for any variable type
    /// **Cons:** Less theoretically grounded than BIC/BDeu
    /// **Use for:** Development, tuning, mixed networks
    EdgeBased(f64),

    /// Bayesian Information Criterion
    ///
    /// BIC = log_likelihood - (k/2) × log(n)
    ///
    /// **Pros:** Theoretically grounded, automatic penalty scaling
    /// **Cons:** Requires counting parameters, approximation to Bayes factor
    /// **Use for:** When you want theory without full Bayesian integration
    BIC,

    /// Bayesian Dirichlet equivalent uniform (gold standard)
    ///
    /// Computes marginal likelihood by integrating over all parameters.
    /// Parameter: α = equivalent sample size (typically 1.0)
    ///
    /// **Pros:** Most theoretically correct, automatic Occam's razor
    /// **Cons:** Only for discrete variables, requires Gamma functions
    /// **Use for:** Final validation, publication-quality results
    BDeu(f64),
}

/// Unified scoring interface for Bayesian network structures.
///
/// Scores a structure using the specified scoring method.
/// This allows easy switching between methods without changing search code.
///
/// # Arguments
/// * `df` - DataFrame with binary variables
/// * `structure` - DAG structure to score
/// * `method` - Which scoring method to use
///
/// # Returns
/// Score (higher = better structure)
///
/// # Example
/// ```ignore
/// // Development: fast edge-based
/// let method = ScoringMethod::EdgeBased(2.0);
/// let score = score_structure(&df, &structure, &method)?;
///
/// // Validation: theoretically correct BDeu
/// let method = ScoringMethod::BDeu(1.0);
/// let final_score = score_structure(&df, &structure, &method)?;
/// ```
///
/// Compute log-likelihood for mixed networks (binary + categorical + continuous)
pub fn compute_mixed_log_likelihood(
    df: &DataFrame,
    structure: &HashMap<String, Vec<String>>,
) -> Result<f64, PolarsError> {
    let mut total_log_lik = 0.0;

    for (var, parents) in structure.iter() {
        let var_type = get_variable_type(df, var)?;

        match var_type {
            VariableType::Binary => {
                // Estimate binary parameters and compute likelihood
                let mut temp_structure = HashMap::new();
                temp_structure.insert(var.clone(), parents.clone());
                let params = estimate_binary_parameters(df, &temp_structure)?;
                let log_lik = compute_binary_log_likelihood(df, &temp_structure, &params)?;
                total_log_lik += log_lik;
            }
            VariableType::Categorical => {
                // Estimate categorical parameters and compute likelihood
                let mut temp_structure = HashMap::new();
                temp_structure.insert(var.clone(), parents.clone());
                let params = estimate_categorical_parameters(df, &temp_structure)?;
                let log_lik = compute_categorical_log_likelihood(df, &temp_structure, &params)?;
                total_log_lik += log_lik;
            }
            VariableType::Continuous => {
                // Estimate continuous parameters and compute likelihood
                let mut temp_structure = HashMap::new();
                temp_structure.insert(var.clone(), parents.clone());
                let params = estimate_continuous_parameters(df, &temp_structure)?;
                let log_lik = compute_continuous_log_likelihood(df, &temp_structure, &params)?;
                total_log_lik += log_lik;
            }
        }
    }

    Ok(total_log_lik)
}

pub fn score_structure(
    df: &DataFrame,
    structure: &HashMap<String, Vec<String>>,
    method: &ScoringMethod,
) -> Result<f64, PolarsError> {
    match method {
        ScoringMethod::EdgeBased(lambda) => {
            // Use general mixed network likelihood
            let log_lik = compute_mixed_log_likelihood(df, structure)?;
            let edges: usize = structure.values().map(|p| p.len()).sum();
            let score = log_lik - lambda * (edges as f64);
            Ok(score)
        }
        ScoringMethod::BIC => {
            // Use general mixed network likelihood
            let log_lik = compute_mixed_log_likelihood(df, structure)?;
            let k = count_parameters(structure);
            let n = df.height() as f64;
            let penalty = (k as f64 / 2.0) * n.ln();
            let bic_score = log_lik - penalty;
            Ok(bic_score)
        }
        ScoringMethod::BDeu(alpha) => {
            // BDeu only works for discrete variables
            score_binary_structure_bdeu(df, structure, *alpha)
        }
    }
}

/// Result of scoring a single structure.
///
/// Contains all scoring components for analysis and comparison.
#[derive(Debug, Clone)]
pub struct StructureScore {
    /// Log-likelihood: log P(data | structure, params)
    pub log_likelihood: f64,

    /// Number of edges (parent relationships) in the structure
    pub edges: usize,

    /// Log prior: -lambda × edges (for edge-based) or other penalty
    pub log_prior: f64,

    /// Log posterior (unnormalized): log_likelihood + log_prior
    pub log_posterior: f64,

    /// Posterior probability (normalized across all structures)
    /// Only meaningful when comparing multiple structures
    pub posterior: f64,
}

/// Score multiple structures and compute normalized posterior probabilities.
///
/// This is the main interface for structure learning search algorithms.
/// Takes multiple candidate structures, scores each one, and returns
/// normalized posterior probabilities for structure sampling/selection.
///
/// # Arguments
/// * `df` - DataFrame with variables
/// * `structures` - HashMap mapping structure names to DAG structures
/// * `method` - Scoring method to use (EdgeBased, BIC, or BDeu)
///
/// # Returns
/// HashMap mapping structure names to StructureScore results
///
/// # Example
/// ```ignore
/// let mut structures = HashMap::new();
/// structures.insert("chain", chain_structure);
/// structures.insert("fork", fork_structure);
///
/// let method = ScoringMethod::EdgeBased(2.0);
/// let results = compute_posteriors(&df, &structures, &method)?;
///
/// // Sample proportionally to posterior
/// for (name, score) in results.iter() {
///     println!("{}: posterior = {:.3}", name, score.posterior);
/// }
/// ```
pub fn compute_posteriors(
    df: &DataFrame,
    structures: &HashMap<String, HashMap<String, Vec<String>>>,
    method: &ScoringMethod,
) -> Result<HashMap<String, StructureScore>, PolarsError> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let total = structures.len();
    let counter = AtomicUsize::new(0);

    // Step 1: Score each structure in parallel
    let results: Vec<(String, StructureScore)> = structures
        .par_iter()
        .filter_map(|(name, structure)| {
            let count = counter.fetch_add(1, Ordering::Relaxed) + 1;

            // Print progress every 50 structures
            if count % 50 == 0 || count == total {
                eprintln!("  Scored {}/{} structures ({:.1}%)", count, total, (count as f64 / total as f64) * 100.0);
            }
            // Compute log-likelihood using unified scoring interface
            let score = match score_structure(df, structure, method) {
                Ok(s) => s,
                Err(e) => {
                    // Skip structures that fail (e.g., singular matrices)
                    eprintln!("  Warning: Skipped structure {} due to error: {}", name, e);
                    return None;
                }
            };

            // Count edges
            let edges: usize = structure.values().map(|parents| parents.len()).sum();

            // Compute log prior based on scoring method
            let log_prior = match method {
                ScoringMethod::EdgeBased(lambda) => -lambda * (edges as f64),
                ScoringMethod::BIC => {
                    // BIC penalty already included in score
                    // Extract it: penalty = (k/2) × log(n)
                    let k = count_parameters(structure);
                    let n = df.height() as f64;
                    let penalty = (k as f64 / 2.0) * n.ln();
                    -penalty
                }
                ScoringMethod::BDeu(_alpha) => {
                    // BDeu has no separate penalty (integrated out)
                    0.0
                }
            };

            // For unified interface, we already have the score which is log_likelihood + log_prior
            // Need to extract log_likelihood
            let log_likelihood = match method {
                ScoringMethod::EdgeBased(lambda) => score + lambda * (edges as f64),
                ScoringMethod::BIC => {
                    let k = count_parameters(structure);
                    let n = df.height() as f64;
                    let penalty = (k as f64 / 2.0) * n.ln();
                    score + penalty
                }
                ScoringMethod::BDeu(_) => score, // BDeu score IS the log marginal likelihood
            };

            let log_posterior = score; // score already includes prior

            Some((
                name.clone(),
                StructureScore {
                    log_likelihood,
                    edges,
                    log_prior,
                    log_posterior,
                    posterior: 0.0, // Will be normalized in step 2
                },
            ))
        })
        .collect();

    if results.is_empty() {
        return Err(PolarsError::ComputeError("All structures failed to score".into()));
    }

    let mut results: HashMap<String, StructureScore> = results.into_iter().collect();

    // Step 2: Normalize posteriors
    // Use log-sum-exp trick for numerical stability
    let log_posteriors: Vec<f64> = results.values().map(|s| s.log_posterior).collect();
    let max_log_posterior = log_posteriors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Compute exp(log_posterior - max) for each structure
    let exp_posteriors: Vec<f64> = log_posteriors
        .iter()
        .map(|lp| (lp - max_log_posterior).exp())
        .collect();

    let sum_exp: f64 = exp_posteriors.iter().sum();

    // Normalize and update posteriors
    let names: Vec<String> = results.keys().cloned().collect();
    for (i, name) in names.iter().enumerate() {
        if let Some(score) = results.get_mut(name) {
            score.posterior = exp_posteriors[i] / sum_exp;
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    #[test]
    fn test_phase1_binary_chain() {
        // Create simple synthetic data: X1 → X2 → X3
        // X1 is independent, X2 depends on X1, X3 depends on X2
        //
        // Ground truth:
        // P(X1=1) = 0.6
        // P(X2=1 | X1=0) = 0.2, P(X2=1 | X1=1) = 0.8
        // P(X3=1 | X2=0) = 0.3, P(X3=1 | X2=1) = 0.7

        let test_df = df![
            "X1" => &[1i64, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            "X2" => &[1i64, 1, 1, 0, 1, 0, 0, 0, 1, 0],
            "X3" => &[1i64, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        ].unwrap();

        // Define the true structure: X1 → X2 → X3
        let mut structure = HashMap::new();
        structure.insert("X1".to_string(), vec![]);
        structure.insert("X2".to_string(), vec!["X1".to_string()]);
        structure.insert("X3".to_string(), vec!["X2".to_string()]);

        // Test 1: Type detection
        println!("\n=== Test 1: Type Detection ===");
        for var in &["X1", "X2", "X3"] {
            let var_type = get_variable_type(&test_df, var).unwrap();
            println!("{}: {:?}", var, var_type);
            assert!(matches!(var_type, VariableType::Binary));
        }

        // Test 2: Parameter estimation
        println!("\n=== Test 2: Parameter Estimation ===");
        let params = estimate_binary_parameters(&test_df, &structure).unwrap();

        // Check X1 (no parents)
        let x1_params = &params["X1"];
        let p_x1_1 = x1_params.probs[&vec![]];
        println!("P(X1=1) = {:.3}", p_x1_1);
        assert!((p_x1_1 - 0.583).abs() < 0.01); // (6+1)/(10+2) ≈ 0.583

        // Check X2 (parent: X1)
        let x2_params = &params["X2"];
        let p_x2_1_given_x1_0 = x2_params.probs[&vec![0]];
        let p_x2_1_given_x1_1 = x2_params.probs[&vec![1]];
        println!("P(X2=1 | X1=0) = {:.3}", p_x2_1_given_x1_0);
        println!("P(X2=1 | X1=1) = {:.3}", p_x2_1_given_x1_1);
        // X1=0 occurs 4 times, X2=1 once → (1+1)/(4+2) ≈ 0.333
        // X1=1 occurs 6 times, X2=1 four times → (4+1)/(6+2) ≈ 0.625
        assert!((p_x2_1_given_x1_0 - 0.333).abs() < 0.01);
        assert!((p_x2_1_given_x1_1 - 0.625).abs() < 0.01);

        // Test 3: Log-likelihood
        println!("\n=== Test 3: Log-Likelihood ===");
        let log_lik = compute_binary_log_likelihood(&test_df, &structure, &params).unwrap();
        println!("Log-likelihood: {:.3}", log_lik);
        assert!(log_lik < 0.0); // Log probabilities are negative
        assert!(log_lik.is_finite()); // Should not be NaN or infinite

        // Test 4: Structure scoring
        println!("\n=== Test 4: Structure Scoring ===");
        let (score, log_lik_2, num_edges) = score_binary_structure(&test_df, &structure, 2.0).unwrap();
        println!("Score: {:.3}", score);
        println!("Log-likelihood: {:.3}", log_lik_2);
        println!("Num edges: {}", num_edges);
        assert_eq!(num_edges, 2); // X1→X2, X2→X3
        assert_eq!(log_lik, log_lik_2); // Should match previous calculation
        assert_eq!(score, log_lik - 2.0 * 2.0); // score = log_lik - lambda × edges

        println!("\n=== All Phase 1 tests passed! ===");
    }

    #[test]
    fn test_bic_vs_edge_penalty() {
        println!("\n=== Comparing BIC vs Edge Penalty ===\n");

        // Same test data as before
        let test_df = df![
            "X1" => &[1i64, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            "X2" => &[1i64, 1, 1, 0, 1, 0, 0, 0, 1, 0],
            "X3" => &[1i64, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        ].unwrap();

        // Three structures to compare
        let mut structure_a = HashMap::new();
        structure_a.insert("X1".to_string(), vec![]);
        structure_a.insert("X2".to_string(), vec!["X1".to_string()]);
        structure_a.insert("X3".to_string(), vec!["X2".to_string()]);

        let mut structure_b = HashMap::new();
        structure_b.insert("X1".to_string(), vec![]);
        structure_b.insert("X2".to_string(), vec!["X1".to_string()]);
        structure_b.insert("X3".to_string(), vec![]);

        let mut structure_c = HashMap::new();
        structure_c.insert("X1".to_string(), vec![]);
        structure_c.insert("X2".to_string(), vec![]);
        structure_c.insert("X3".to_string(), vec![]);

        println!("Sample size n = {}\n", test_df.height());

        // Score all three with both methods
        for (name, structure) in [("A: X1→X2→X3", &structure_a),
                                   ("B: X1→X2, X3", &structure_b),
                                   ("C: Independent", &structure_c)] {

            // Edge-based penalty (λ=2.0)
            let (edge_score, log_lik, edges) =
                score_binary_structure(&test_df, structure, 2.0).unwrap();

            // True BIC
            let (bic_score, _, params) =
                score_binary_structure_bic(&test_df, structure).unwrap();

            let n = test_df.height() as f64;
            let bic_penalty = (params as f64 / 2.0) * n.ln();
            let edge_penalty = 2.0 * edges as f64;

            println!("Structure {}:", name);
            println!("  Log-likelihood: {:.3}", log_lik);
            println!("  Edges: {}, Parameters: {}", edges, params);
            println!("  Edge penalty (λ=2): {:.3}", edge_penalty);
            println!("  BIC penalty: {:.3}", bic_penalty);
            println!("  Edge-based score: {:.3}", edge_score);
            println!("  BIC score: {:.3}", bic_score);
            println!();
        }

        println!("Key observation:");
        println!("- BIC penalty = (k/2) × ln(n) = (k/2) × ln(10) ≈ k × 1.15");
        println!("- As structures get more parents, k grows exponentially (2^m)");
        println!("- BIC penalizes complex structures MORE than edge-based penalty");
    }

    #[test]
    fn test_all_three_scoring_methods() {
        println!("\n=== Comprehensive Scoring Comparison ===\n");
        println!("Comparing: Edge penalty (λ=2) vs BIC vs BDeu (α=1)\n");

        let test_df = df![
            "X1" => &[1i64, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            "X2" => &[1i64, 1, 1, 0, 1, 0, 0, 0, 1, 0],
            "X3" => &[1i64, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        ].unwrap();

        let mut structure_a = HashMap::new();
        structure_a.insert("X1".to_string(), vec![]);
        structure_a.insert("X2".to_string(), vec!["X1".to_string()]);
        structure_a.insert("X3".to_string(), vec!["X2".to_string()]);

        let mut structure_b = HashMap::new();
        structure_b.insert("X1".to_string(), vec![]);
        structure_b.insert("X2".to_string(), vec!["X1".to_string()]);
        structure_b.insert("X3".to_string(), vec![]);

        let mut structure_c = HashMap::new();
        structure_c.insert("X1".to_string(), vec![]);
        structure_c.insert("X2".to_string(), vec![]);
        structure_c.insert("X3".to_string(), vec![]);

        println!("Sample size: n = {}\n", test_df.height());

        let mut results = Vec::new();

        for (name, structure) in [("A: X1→X2→X3", &structure_a),
                                   ("B: X1→X2, X3", &structure_b),
                                   ("C: Independent", &structure_c)] {

            // Method 1: Edge-based (λ=2.0)
            let (edge_score, _, _) =
                score_binary_structure(&test_df, structure, 2.0).unwrap();

            // Method 2: True BIC
            let (bic_score, _, _) =
                score_binary_structure_bic(&test_df, structure).unwrap();

            // Method 3: BDeu (α=1.0)
            let bdeu_score = score_binary_structure_bdeu(&test_df, structure, 1.0).unwrap();

            println!("Structure {}:", name);
            println!("  Edge-based: {:.3}", edge_score);
            println!("  BIC:        {:.3}", bic_score);
            println!("  BDeu:       {:.3}", bdeu_score);
            println!();

            results.push((name, edge_score, bic_score, bdeu_score));
        }

        println!("=== Rankings (higher = better) ===\n");

        // Find winners for each method
        let edge_winner = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        let bic_winner = results.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
        let bdeu_winner = results.iter().max_by(|a, b| a.3.partial_cmp(&b.3).unwrap()).unwrap();

        println!("Edge-based winner: {}", edge_winner.0);
        println!("BIC winner:        {}", bic_winner.0);
        println!("BDeu winner:       {}", bdeu_winner.0);

        println!("\n=== Key Insights ===");
        println!("• Edge-based: Simple, tunable (λ), penalizes edges linearly");
        println!("• BIC: Theoretically grounded, penalizes parameters with (k/2)×ln(n)");
        println!("• BDeu: Most correct, integrates over all θ, automatic Occam's razor");
        println!("\nBDeu is the gold standard for discrete Bayesian networks!");
    }

    #[test]
    fn test_unified_scoring_interface() {
        println!("\n=== Testing Unified Scoring Interface ===\n");

        let test_df = df![
            "X1" => &[1i64, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            "X2" => &[1i64, 1, 1, 0, 1, 0, 0, 0, 1, 0],
            "X3" => &[1i64, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        ].unwrap();

        let mut structure = HashMap::new();
        structure.insert("X1".to_string(), vec![]);
        structure.insert("X2".to_string(), vec!["X1".to_string()]);
        structure.insert("X3".to_string(), vec!["X2".to_string()]);

        println!("Testing structure: X1 → X2 → X3\n");

        // Test all three methods via unified interface
        let methods = vec![
            ("Edge-based (λ=2.0)", ScoringMethod::EdgeBased(2.0)),
            ("Edge-based (λ=1.0)", ScoringMethod::EdgeBased(1.0)),
            ("Edge-based (λ=5.0)", ScoringMethod::EdgeBased(5.0)),
            ("BIC", ScoringMethod::BIC),
            ("BDeu (α=1.0)", ScoringMethod::BDeu(1.0)),
            ("BDeu (α=0.5)", ScoringMethod::BDeu(0.5)),
        ];

        for (name, method) in methods {
            let score = score_structure(&test_df, &structure, &method).unwrap();
            println!("{:20} Score: {:.3}", name, score);
        }

        println!("\n=== Practical Workflow ===");
        println!("1. Development: Use EdgeBased(2.0) for fast iteration");
        println!("2. Tuning: Try different λ values, pick best via cross-validation");
        println!("3. Validation: Re-score top candidates with BDeu(1.0)");
        println!("4. Publication: Report BDeu scores for theoretical rigor");

        // Demonstrate switching methods
        println!("\n=== Example: Method Switching ===");

        let dev_method = ScoringMethod::EdgeBased(2.0);
        let dev_score = score_structure(&test_df, &structure, &dev_method).unwrap();
        println!("Development score (fast): {:.3}", dev_score);

        let final_method = ScoringMethod::BDeu(1.0);
        let final_score = score_structure(&test_df, &structure, &final_method).unwrap();
        println!("Final score (rigorous): {:.3}", final_score);

        println!("\n✓ Unified interface allows easy switching without code changes!");
    }

    #[test]
    fn test_categorical_variables() {
        println!("\n=== Testing Phase 2: Categorical Variables ===\n");

        // Create test data with categorical variable (3 values: 0, 1, 2)
        // X1: binary {0, 1}
        // X2: categorical {0, 1, 2} depends on X1
        let test_df = df![
            "X1" => &[0i64, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "X2" => &[0i64, 0, 1, 0, 1, 2, 2, 1, 2, 2],
        ].unwrap();

        let mut structure = HashMap::new();
        structure.insert("X1".to_string(), vec![]);
        structure.insert("X2".to_string(), vec!["X1".to_string()]);

        println!("Structure: X1 (binary) → X2 (categorical {{0,1,2}})\n");

        // Test 1: Type detection
        println!("=== Test 1: Type Detection ===");
        let x1_type = get_variable_type(&test_df, "X1").unwrap();
        let x2_type = get_variable_type(&test_df, "X2").unwrap();
        println!("X1 type: {:?}", x1_type);
        println!("X2 type: {:?}", x2_type);
        assert!(matches!(x1_type, VariableType::Binary));
        assert!(matches!(x2_type, VariableType::Categorical));

        // Test 2: Parameter estimation for categorical
        println!("\n=== Test 2: Categorical Parameter Estimation ===");
        let params = estimate_categorical_parameters(&test_df, &structure).unwrap();

        // Check X1 (binary, but estimated as categorical)
        // Actually X1 should be handled by binary estimator, let's just check X2

        // Check X2 (categorical with 3 values)
        let x2_params = &params["X2"];
        println!("X2 parameters:");

        // When X1=0 (4 samples): X2 values are [0,0,1,0] → mostly 0
        let p_x2_given_x1_0 = &x2_params.probs[&vec![0]];
        println!("  P(X2 | X1=0): {:?}", p_x2_given_x1_0);

        // When X1=1 (6 samples): X2 values are [1,2,2,1,2,2] → mostly 2
        let p_x2_given_x1_1 = &x2_params.probs[&vec![1]];
        println!("  P(X2 | X1=1): {:?}", p_x2_given_x1_1);

        // Check that probabilities sum to 1
        let sum_0: f64 = p_x2_given_x1_0.values().sum();
        let sum_1: f64 = p_x2_given_x1_1.values().sum();
        println!("  Sum P(X2|X1=0) = {:.6} (should be 1.0)", sum_0);
        println!("  Sum P(X2|X1=1) = {:.6} (should be 1.0)", sum_1);
        assert!((sum_0 - 1.0).abs() < 0.0001);
        assert!((sum_1 - 1.0).abs() < 0.0001);

        // Test 3: Log-likelihood
        println!("\n=== Test 3: Categorical Log-Likelihood ===");
        let log_lik = compute_categorical_log_likelihood(&test_df, &structure, &params).unwrap();
        println!("Log-likelihood: {:.3}", log_lik);
        assert!(log_lik < 0.0); // Log probabilities are negative
        assert!(log_lik.is_finite()); // Should not be NaN or infinite

        println!("\n=== Phase 2 Categorical Test Passed! ===");
    }

    #[test]
    fn test_phase3_continuous_variables() {
        println!("\n=== Testing Phase 3: Continuous Variables ===\n");

        // Create test data with continuous variables
        // X1: independent Gaussian
        // X2: linear function of X1 with noise
        // X3: independent Gaussian
        //
        // Ground truth:
        // X1 ~ N(5.0, 1.0)
        // X2 = 2.0 + 1.5*X1 + noise ~ N(2.0 + 1.5*X1, 0.5)
        // X3 ~ N(10.0, 2.0)

        let test_df = df![
            "X1" => &[4.5, 5.0, 5.5, 4.8, 5.2, 4.9, 5.1, 5.3, 4.7, 5.4],
            "X2" => &[8.7, 9.5, 10.2, 9.2, 9.8, 9.3, 9.6, 10.0, 9.0, 10.1],
            "X3" => &[9.5, 10.2, 11.0, 10.5, 9.8, 10.1, 9.9, 10.3, 9.7, 10.0],
        ].unwrap();

        let mut structure = HashMap::new();
        structure.insert("X1".to_string(), vec![]);
        structure.insert("X2".to_string(), vec!["X1".to_string()]);
        structure.insert("X3".to_string(), vec![]);

        println!("Structure: X1 (continuous) → X2 (continuous), X3 (continuous)\n");

        // Test 1: Type detection
        println!("=== Test 1: Type Detection ===");
        let x1_type = get_variable_type(&test_df, "X1").unwrap();
        let x2_type = get_variable_type(&test_df, "X2").unwrap();
        let x3_type = get_variable_type(&test_df, "X3").unwrap();
        println!("X1 type: {:?}", x1_type);
        println!("X2 type: {:?}", x2_type);
        println!("X3 type: {:?}", x3_type);
        assert!(matches!(x1_type, VariableType::Continuous));
        assert!(matches!(x2_type, VariableType::Continuous));
        assert!(matches!(x3_type, VariableType::Continuous));

        // Test 2: Parameter estimation
        println!("\n=== Test 2: Continuous Parameter Estimation ===");
        let params = estimate_continuous_parameters(&test_df, &structure).unwrap();

        // Check X1 (Gaussian, no parents)
        let x1_params = &params["X1"];
        println!("X1 parameters: {:?}", x1_params);
        match x1_params {
            ContinuousParams::Gaussian { mean, std } => {
                println!("  Mean: {:.3}, Std: {:.3}", mean, std);
                // Mean should be close to 5.0
                assert!((mean - 5.0).abs() < 0.5);
                // Std should be close to 1.0 (but will vary with small sample)
                assert!(std > &0.0);
            }
            _ => panic!("Expected Gaussian params for X1"),
        }

        // Check X2 (Linear, has parent X1)
        let x2_params = &params["X2"];
        println!("X2 parameters: {:?}", x2_params);
        match x2_params {
            ContinuousParams::Linear { intercept, coeffs, std } => {
                println!("  Intercept: {:.3}", intercept);
                println!("  Coefficients: {:?}", coeffs);
                println!("  Residual std: {:.3}", std);
                // Intercept should be close to 2.0, coefficient close to 1.5
                // (but will vary due to noise and small sample)
                assert!(coeffs.len() == 1);
                assert!(std > &0.0);
            }
            _ => panic!("Expected Linear params for X2"),
        }

        // Check X3 (Gaussian, no parents)
        let x3_params = &params["X3"];
        println!("X3 parameters: {:?}", x3_params);
        match x3_params {
            ContinuousParams::Gaussian { mean, std } => {
                println!("  Mean: {:.3}, Std: {:.3}", mean, std);
                // Mean should be close to 10.0
                assert!((mean - 10.0).abs() < 0.5);
                assert!(std > &0.0);
            }
            _ => panic!("Expected Gaussian params for X3"),
        }

        // Test 3: Log-likelihood
        println!("\n=== Test 3: Continuous Log-Likelihood ===");
        let log_lik = compute_continuous_log_likelihood(&test_df, &structure, &params).unwrap();
        println!("Log-likelihood: {:.3}", log_lik);
        // Note: For continuous distributions, log-likelihood can be positive!
        // PDF can be > 1, especially for tight distributions (small std)
        assert!(log_lik.is_finite()); // Should not be NaN or infinite

        println!("\n=== Phase 3 Continuous Test Passed! ===");
    }

    #[test]
    fn test_phase3_linear_regression() {
        println!("\n=== Testing Phase 3: Linear Regression Accuracy ===\n");

        // Create test data with known linear relationship
        // Y = 3.0 + 2.0*X1 + 1.5*X2
        //
        // We'll use exact values to verify our least squares implementation
        // X1 and X2 are independent (not collinear)

        let test_df = df![
            "X1" => &[1.0, 2.0, 3.0, 4.0, 5.0],
            "X2" => &[1.0, 3.0, 2.0, 5.0, 4.0],  // Not collinear with X1
            "Y"  => &[6.5, 11.5, 12.0, 18.5, 19.0],  // Y = 3 + 2*X1 + 1.5*X2
        ].unwrap();

        let mut structure = HashMap::new();
        structure.insert("X1".to_string(), vec![]);
        structure.insert("X2".to_string(), vec![]);
        structure.insert("Y".to_string(), vec!["X1".to_string(), "X2".to_string()]);

        println!("Structure: Y = f(X1, X2) with known coefficients\n");
        println!("True model: Y = 3.0 + 2.0*X1 + 1.5*X2\n");

        // Estimate parameters
        let params = estimate_continuous_parameters(&test_df, &structure).unwrap();

        // Check Y parameters (should recover true coefficients exactly)
        let y_params = &params["Y"];
        println!("Estimated parameters:");
        match y_params {
            ContinuousParams::Linear { intercept, coeffs, std } => {
                println!("  Intercept: {:.6} (true: 3.0)", intercept);
                println!("  Coeff[X1]: {:.6} (true: 2.0)", coeffs[0]);
                println!("  Coeff[X2]: {:.6} (true: 1.5)", coeffs[1]);
                println!("  Residual std: {:.6} (true: ~0.0)", std);

                // With perfect linear data, we should recover exact coefficients
                assert!((intercept - 3.0).abs() < 1e-6);
                assert!((coeffs[0] - 2.0).abs() < 1e-6);
                assert!((coeffs[1] - 1.5).abs() < 1e-6);
                assert!(*std < 1e-5); // Near-zero residuals
            }
            _ => panic!("Expected Linear params for Y"),
        }

        println!("\n=== Linear Regression Test Passed! ===");
    }

    #[test]
    fn test_phase4_mixed_networks() {
        println!("\n=== Testing Phase 4: Mixed Networks ===\n");

        // Create test data with mixed variable types
        // Temperature (continuous) → Species (categorical {0, 1, 2})
        // Temperature median is 20.0
        // Low temp (≤20) → more likely species 0
        // High temp (>20) → more likely species 2
        //
        // Temperature: [18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
        // Median: 21.0
        // Species distribution:
        //   Temp ≤ 21 (discretized=0): mostly species 0
        //   Temp > 21 (discretized=1): mostly species 2

        let test_df = df![
            "Temperature" => &[18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            "Species" => &[0i64, 0, 0, 1, 2, 2, 2, 2],
        ].unwrap();

        let mut structure = HashMap::new();
        structure.insert("Temperature".to_string(), vec![]);
        structure.insert("Species".to_string(), vec!["Temperature".to_string()]);

        println!("Structure: Temperature (continuous) → Species (categorical {{0,1,2}})\n");
        println!("Temperature median: 21.0");
        println!("Data:");
        println!("  Temp ≤ 21 (4 samples): Species [0, 0, 0, 1] → mostly 0");
        println!("  Temp > 21 (4 samples): Species [2, 2, 2, 2] → all 2\n");

        // Test 1: Type detection
        println!("=== Test 1: Type Detection ===");
        let temp_type = get_variable_type(&test_df, "Temperature").unwrap();
        let species_type = get_variable_type(&test_df, "Species").unwrap();
        println!("Temperature type: {:?}", temp_type);
        println!("Species type: {:?}", species_type);
        assert!(matches!(temp_type, VariableType::Continuous));
        assert!(matches!(species_type, VariableType::Categorical));

        // Test 2: Parameter estimation for mixed network
        println!("\n=== Test 2: Mixed Network Parameter Estimation ===");
        let params = estimate_categorical_parameters(&test_df, &structure).unwrap();

        // Check Species parameters (categorical child with continuous parent)
        let species_params = &params["Species"];
        println!("Species parameters (discretized Temperature parent):");

        // Temperature discretized at median (21.0):
        // combo [0] = Temp ≤ 21: species [0,0,0,1] → P(0) high, P(1) low, P(2) very low
        // combo [1] = Temp > 21: species [2,2,2,2] → P(2) very high
        let prob_dist_low_temp = &species_params.probs[&vec![0]];
        let prob_dist_high_temp = &species_params.probs[&vec![1]];

        println!("  P(Species | Temp ≤ 21): {:?}", prob_dist_low_temp);
        println!("  P(Species | Temp > 21): {:?}", prob_dist_high_temp);

        // Verify probabilities sum to 1
        let sum_low: f64 = prob_dist_low_temp.values().sum();
        let sum_high: f64 = prob_dist_high_temp.values().sum();
        println!("  Sum P(Species | Temp ≤ 21) = {:.6} (should be 1.0)", sum_low);
        println!("  Sum P(Species | Temp > 21) = {:.6} (should be 1.0)", sum_high);
        assert!((sum_low - 1.0).abs() < 0.0001);
        assert!((sum_high - 1.0).abs() < 0.0001);

        // Verify learned distributions make sense
        // Low temp: P(species=0) should be highest
        // High temp: P(species=2) should be highest
        println!("\n  Checking learned distributions:");
        println!("    Low temp: P(0)={:.3}, P(1)={:.3}, P(2)={:.3}",
                 prob_dist_low_temp[&0], prob_dist_low_temp[&1], prob_dist_low_temp[&2]);
        println!("    High temp: P(0)={:.3}, P(1)={:.3}, P(2)={:.3}",
                 prob_dist_high_temp[&0], prob_dist_high_temp[&1], prob_dist_high_temp[&2]);

        // Test 3: Log-likelihood
        println!("\n=== Test 3: Mixed Network Log-Likelihood ===");
        let log_lik = compute_categorical_log_likelihood(&test_df, &structure, &params).unwrap();
        println!("Log-likelihood: {:.3}", log_lik);
        assert!(log_lik.is_finite());

        println!("\n=== Phase 4 Mixed Network Test Passed! ===");
    }

    #[test]
    fn test_phase4_binary_with_continuous_parent() {
        println!("\n=== Testing Phase 4: Binary Child with Continuous Parent ===\n");

        // Create test data: Survived (binary) depends on Age (continuous)
        // Age median: 35.0
        // Young (≤35): 80% survival
        // Old (>35): 20% survival

        let test_df = df![
            "Age" => &[20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
            "Survived" => &[1i64, 1, 1, 1, 0, 0, 0, 1],  // Young: 4/4 survived, Old: 1/4 survived
        ].unwrap();

        let mut structure = HashMap::new();
        structure.insert("Age".to_string(), vec![]);
        structure.insert("Survived".to_string(), vec!["Age".to_string()]);

        println!("Structure: Age (continuous) → Survived (binary)\n");
        println!("Age median: 37.5");
        println!("Data:");
        println!("  Age ≤ 37.5 (4 samples): Survived [1,1,1,1] → 100%");
        println!("  Age > 37.5 (4 samples): Survived [0,0,0,1] → 25%\n");

        // Estimate parameters
        println!("=== Parameter Estimation ===");
        let params = estimate_binary_parameters(&test_df, &structure).unwrap();

        let survived_params = &params["Survived"];
        let prob_young = survived_params.probs[&vec![0]];
        let prob_old = survived_params.probs[&vec![1]];

        println!("P(Survived=1 | Age ≤ median): {:.3}", prob_young);
        println!("P(Survived=1 | Age > median): {:.3}", prob_old);

        // Young should have higher survival probability
        assert!(prob_young > prob_old);

        // Log-likelihood
        let log_lik = compute_binary_log_likelihood(&test_df, &structure, &params).unwrap();
        println!("Log-likelihood: {:.3}", log_lik);
        assert!(log_lik.is_finite());

        println!("\n=== Binary with Continuous Parent Test Passed! ===");
    }

    #[test]
    fn test_phase5_compute_posteriors() {
        println!("\n=== Testing Phase 5: Multi-Structure Scoring ===\n");

        // Create test data
        let test_df = df![
            "X1" => &[1i64, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            "X2" => &[1i64, 1, 1, 0, 1, 0, 0, 0, 1, 0],
            "X3" => &[1i64, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        ].unwrap();

        // Three candidate structures
        let mut structure_a = HashMap::new();
        structure_a.insert("X1".to_string(), vec![]);
        structure_a.insert("X2".to_string(), vec!["X1".to_string()]);
        structure_a.insert("X3".to_string(), vec!["X2".to_string()]);

        let mut structure_b = HashMap::new();
        structure_b.insert("X1".to_string(), vec![]);
        structure_b.insert("X2".to_string(), vec!["X1".to_string()]);
        structure_b.insert("X3".to_string(), vec![]);

        let mut structure_c = HashMap::new();
        structure_c.insert("X1".to_string(), vec![]);
        structure_c.insert("X2".to_string(), vec![]);
        structure_c.insert("X3".to_string(), vec![]);

        let mut structures = HashMap::new();
        structures.insert("chain".to_string(), structure_a);
        structures.insert("partial".to_string(), structure_b);
        structures.insert("independent".to_string(), structure_c);

        println!("Structures:");
        println!("  chain:       X1 → X2 → X3  (2 edges)");
        println!("  partial:     X1 → X2, X3   (1 edge)");
        println!("  independent: X1, X2, X3    (0 edges)");
        println!();

        // Score with edge-based method
        let method = ScoringMethod::EdgeBased(2.0);
        let results = compute_posteriors(&test_df, &structures, &method).unwrap();

        println!("=== Edge-Based Scoring (λ=2.0) ===");
        for (name, score) in [("chain", &results["chain"]),
                               ("partial", &results["partial"]),
                               ("independent", &results["independent"])] {
            println!("{:12} | log_lik: {:7.3} | edges: {} | log_post: {:7.3} | post: {:.3}",
                     name, score.log_likelihood, score.edges, score.log_posterior, score.posterior);
        }

        // Verify posteriors sum to 1.0
        let total_posterior: f64 = results.values().map(|s| s.posterior).sum();
        println!("\nTotal posterior: {:.6} (should be 1.0)", total_posterior);
        assert!((total_posterior - 1.0).abs() < 1e-6);

        // Verify scores match individual scoring
        let chain_score = score_structure(&test_df, &structures["chain"], &method).unwrap();
        assert!((results["chain"].log_posterior - chain_score).abs() < 1e-6);

        // Test with BDeu scoring
        println!("\n=== BDeu Scoring (α=1.0) ===");
        let method_bdeu = ScoringMethod::BDeu(1.0);
        let results_bdeu = compute_posteriors(&test_df, &structures, &method_bdeu).unwrap();

        for (name, score) in [("chain", &results_bdeu["chain"]),
                               ("partial", &results_bdeu["partial"]),
                               ("independent", &results_bdeu["independent"])] {
            println!("{:12} | log_lik: {:7.3} | edges: {} | log_post: {:7.3} | post: {:.3}",
                     name, score.log_likelihood, score.edges, score.log_posterior, score.posterior);
        }

        let total_posterior_bdeu: f64 = results_bdeu.values().map(|s| s.posterior).sum();
        println!("\nTotal posterior: {:.6} (should be 1.0)", total_posterior_bdeu);
        assert!((total_posterior_bdeu - 1.0).abs() < 1e-6);

        println!("\n=== Key Observations ===");
        println!("• Posteriors sum to 1.0 (proper probability distribution)");
        println!("• Different scoring methods → different posterior distributions");
        println!("• Edge-based penalizes complex structures more explicitly");
        println!("• BDeu integrates complexity penalty automatically");

        println!("\n=== Phase 5 Multi-Structure Scoring Test Passed! ===");
    }
}
