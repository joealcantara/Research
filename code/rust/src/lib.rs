pub use ndarray::Array2;

pub fn compute_log_likelihood_continuous(
    data: &Array2<f64>,
    var: usize,
    parents: &[usize],
    is_categorical: bool,
) -> f64 {
    use ndarray::Array2;
    use std::f64::consts::PI;

    let y = data.column(var);
    let n = y.len();

    if parents.is_empty() {
        // No parents: marginal Gaussian
        if is_categorical {
            // Categorical marginal (simplified - just count probabilities)
            let unique_vals = vec![1.0, 2.0, 3.0];
            let mut log_lik = 0.0;
            for &val in &unique_vals {
                let count = y.iter().filter(|&&v| v == val).count() as f64;
                let prob = (count + 1.0) / (n as f64 + unique_vals.len() as f64);
                log_lik += count * prob.ln();
            }
            log_lik
        } else {
            let mean = y.mean().unwrap();
            let std = y.std(0.0) + 1e-6;
            -0.5 * n as f64 * (2.0 * PI * std * std).ln()
                - 0.5 * y.iter().map(|&yi| ((yi - mean) / std).powi(2)).sum::<f64>()
        }
    } else {
        // Has parents
        if is_categorical {
            // Categorical with continuous parents: discretize parents at median, compute CPT
            use itertools::Itertools;

            let unique_vals = vec![1.0, 2.0, 3.0];
            let num_values = unique_vals.len();

            // Discretize continuous parents at median
            let mut parent_bins: Vec<Vec<usize>> = Vec::new();
            for &parent_idx in parents {
                let parent_col = data.column(parent_idx);
                let median = {
                    let mut sorted: Vec<f64> = parent_col.iter().copied().collect();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    sorted[sorted.len() / 2]
                };

                let bins: Vec<usize> = parent_col.iter()
                    .map(|&val| if val <= median { 0 } else { 1 })
                    .collect();
                parent_bins.push(bins);
            }

            // Generate all combinations of parent values (0 or 1 for each parent)
            let n_parents = parents.len();
            let all_combos: Vec<Vec<usize>> = (0..n_parents)
                .map(|_| vec![0, 1])
                .multi_cartesian_product()
                .collect();

            let mut log_lik = 0.0;

            for combo in all_combos {
                // Find rows where parents match this combination
                let mut mask = vec![true; n];
                for (parent_idx, &bin_val) in combo.iter().enumerate() {
                    for row in 0..n {
                        if parent_bins[parent_idx][row] != bin_val {
                            mask[row] = false;
                        }
                    }
                }

                let count_total = mask.iter().filter(|&&m| m).count();
                if count_total == 0 {
                    continue;
                }

                // For each unique value, compute P(val | parent_combo)
                for &val in &unique_vals {
                    let mut count_val = 0;
                    for row in 0..n {
                        if mask[row] && y[row] == val {
                            count_val += 1;
                        }
                    }

                    let prob = (count_val as f64 + 1.0) / (count_total as f64 + num_values as f64);
                    log_lik += count_val as f64 * prob.ln();
                }
            }

            log_lik
        } else {
            // Linear regression: y = X*beta + intercept + noise
            let p = parents.len();
            let mut x = Array2::zeros((n, p + 1));
            x.column_mut(0).fill(1.0); // Intercept

            for (i, &parent) in parents.iter().enumerate() {
                x.column_mut(i + 1).assign(&data.column(parent));
            }

            // Solve least squares: beta = (X^T X)^{-1} X^T y
            let xt = x.t();
            let xtx = xt.dot(&x);
            let xty = xt.dot(&y.to_owned());

            // Simple linear solve (pseudo-inverse approach)
            match ndarray_linalg::solve::Inverse::inv(&xtx) {
                Ok(xtx_inv) => {
                    let beta = xtx_inv.dot(&xty);
                    let predictions = x.dot(&beta);
                    let residuals = &y.to_owned() - &predictions;
                    let residual_std = residuals.std(0.0) + 1e-6;

                    -0.5 * n as f64 * (2.0 * PI * residual_std * residual_std).ln()
                        - 0.5 * residuals.iter().map(|&r| (r / residual_std).powi(2)).sum::<f64>()
                }
                Err(_) => {
                    // Fallback to marginal if regression fails
                    let mean = y.mean().unwrap();
                    let std = y.std(0.0) + 1e-6;
                    -0.5 * n as f64 * (2.0 * PI * std * std).ln()
                        - 0.5 * y.iter().map(|&yi| ((yi - mean) / std).powi(2)).sum::<f64>()
                }
            }
        }
    }
}
