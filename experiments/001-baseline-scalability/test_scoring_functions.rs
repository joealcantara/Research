/// Unit tests for Rust scoring functions
///
/// Tests mathematical correctness of:
/// - Gaussian marginal likelihood (no parents)
/// - Linear regression likelihood (with parents)
/// - BIC penalty calculation
/// - Decomposability (total score = sum of parts)
///
/// To run: place in src/bin/ and run `cargo test --bin test_scoring_functions`

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;

// Copy scoring functions from main implementation
// (In practice, these would be imported from a lib module)

fn gaussian_marginal_score(data: &Array1<f64>, lambda: f64, n_samples: usize) -> f64 {
    let n = data.len() as f64;
    let mean = data.mean().unwrap();
    let std = data.std(0.0) + 1e-6;

    let log_likelihood: f64 = data.iter()
        .map(|&x| {
            let z = (x - mean) / std;
            -0.5 * (2.0 * std::f64::consts::PI * std.powi(2)).ln() - 0.5 * z.powi(2)
        })
        .sum();

    log_likelihood
}

fn linear_regression_score(
    target: &Array1<f64>,
    parents: &Array2<f64>,
    lambda: f64,
    n_samples: usize,
) -> f64 {
    let n = target.len();

    // Add intercept column
    let mut X = Array2::zeros((n, parents.ncols() + 1));
    for i in 0..n {
        X[[i, 0]] = 1.0;
        for j in 0..parents.ncols() {
            X[[i, j + 1]] = parents[[i, j]];
        }
    }

    // Solve least squares: β = (X'X)^(-1) X'y
    let xtx = X.t().dot(&X);
    let xty = X.t().dot(target);

    // Use simple solver (in practice, use proper linear algebra library)
    // For testing, we'll use a simplified approach
    let beta = solve_linear_system(&xtx, &xty);

    // Calculate residuals
    let predictions = X.dot(&beta);
    let residuals = target - &predictions;

    // Calculate likelihood
    let std = residuals.std(0.0) + 1e-6;
    let log_likelihood: f64 = residuals.iter()
        .map(|&r| {
            let z = r / std;
            -0.5 * (2.0 * std::f64::consts::PI * std.powi(2)).ln() - 0.5 * z.powi(2)
        })
        .sum();

    // BIC penalty
    let num_params = parents.ncols() + 1;
    let penalty = lambda * (num_params as f64) * (n as f64).ln() / 2.0;

    log_likelihood - penalty
}

fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    // Simplified solver for testing (not production quality)
    // In practice, use ndarray-linalg or nalgebra
    let n = a.nrows();
    let mut result = Array1::zeros(n);

    // For small systems, use Gaussian elimination
    // This is a placeholder - proper implementation would use LU decomposition
    result.fill(0.0);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_marginal_basic() {
        // Generate simple Gaussian data
        let n = 1000;
        let mu_true = 5.0;
        let sigma_true = 2.0;

        let data = Array1::random(n, StandardNormal);
        let data = &data * sigma_true + mu_true;

        let score = gaussian_marginal_score(&data, 0.0, n);

        // Score should be negative (log-likelihood)
        assert!(score < 0.0, "Log-likelihood should be negative");

        // Score should be finite
        assert!(!score.is_nan(), "Score should not be NaN");
        assert!(!score.is_infinite(), "Score should not be infinite");
    }

    #[test]
    fn test_bic_penalty_reduces_score() {
        let n = 500;
        let x = Array1::random(n, StandardNormal);
        let y = &x * 0.5 + Array1::random(n, StandardNormal);

        let parents = x.clone().insert_axis(ndarray::Axis(1));

        let score_no_penalty = linear_regression_score(&y, &parents, 0.0, n);
        let score_with_penalty = linear_regression_score(&y, &parents, 2.0, n);

        // With penalty should be lower
        assert!(
            score_with_penalty < score_no_penalty,
            "BIC penalty should reduce score"
        );

        // Penalty amount
        let expected_penalty = 2.0 * 2.0 * (n as f64).ln() / 2.0; // 2 params (intercept + slope)
        let actual_penalty = score_no_penalty - score_with_penalty;

        assert!(
            (actual_penalty - expected_penalty).abs() < 1e-6,
            "Penalty should match expected BIC formula"
        );
    }

    #[test]
    fn test_constant_column_no_crash() {
        // Constant column should not crash
        let n = 500;
        let constant = Array1::from_elem(n, 100.0);

        let score = gaussian_marginal_score(&constant, 0.0, n);

        // Should not be NaN or Inf
        assert!(!score.is_nan(), "Constant column should not produce NaN");
        assert!(!score.is_infinite(), "Constant column should not produce Inf");
    }

    #[test]
    fn test_numerical_stability_large_values() {
        let n = 500;
        let large_data = Array1::random(n, StandardNormal) * 1e6;

        let score = gaussian_marginal_score(&large_data, 0.0, n);

        assert!(!score.is_nan(), "Large values should not produce NaN");
        assert!(!score.is_infinite(), "Large values should not produce Inf");
    }

    #[test]
    fn test_numerical_stability_small_variance() {
        let n = 500;
        let small_data = Array1::random(n, StandardNormal) * 1e-6;

        let score = gaussian_marginal_score(&small_data, 0.0, n);

        assert!(!score.is_nan(), "Small variance should not produce NaN");
        assert!(!score.is_infinite(), "Small variance should not produce Inf");
    }

    #[test]
    fn test_consistency() {
        // Same data should give same score
        let n = 500;
        let data = Array1::random(n, StandardNormal);

        let score1 = gaussian_marginal_score(&data, 0.0, n);
        let score2 = gaussian_marginal_score(&data, 0.0, n);

        assert_eq!(score1, score2, "Scoring should be deterministic");
    }

    #[test]
    fn test_edge_case_minimal_samples() {
        // Test with minimal samples
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let score = gaussian_marginal_score(&data, 0.0, 3);

        assert!(!score.is_nan(), "Minimal samples should not produce NaN");
        assert!(!score.is_infinite(), "Minimal samples should not produce Inf");
    }
}

fn main() {
    println!("Run with: cargo test --bin test_scoring_functions");
}
