use ndarray::Array2;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const PI: f64 = std::f64::consts::PI;

/// Load SECOM dataset from CSV
fn load_secom_data(n_features: usize) -> (Array2<f64>, usize, usize) {
    let home = env::var("HOME").unwrap();
    let data_path = format!(
        "{}/Documents/research/experiments/001-baseline-scalability/subsets/secom_{}.csv",
        home, n_features
    );

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&data_path)
        .expect("Failed to open SECOM data file");

    let mut data = Vec::new();
    let mut n_cols = 0;

    for result in reader.records() {
        let record = result.unwrap();
        let row: Vec<f64> = record.iter().map(|s| s.parse().unwrap()).collect();
        n_cols = row.len();
        data.extend(row);
    }

    let n_rows = data.len() / n_cols;
    (
        Array2::from_shape_vec((n_rows, n_cols), data).unwrap(),
        n_rows,
        n_cols,
    )
}

/// Compute log-likelihood for continuous variable with linear regression
fn compute_log_likelihood(data: &Array2<f64>, var: usize, parents: &[usize]) -> f64 {
    let y = data.column(var);
    let n = y.len();

    if parents.is_empty() {
        // No parents: marginal Gaussian
        let mean = y.mean().unwrap();
        let std = y.std(0.0) + 1e-6;
        -0.5 * n as f64 * (2.0 * PI * std * std).ln()
            - 0.5 * y.iter().map(|&yi| ((yi - mean) / std).powi(2)).sum::<f64>()
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

        match ndarray_linalg::solve::Inverse::inv(&xtx) {
            Ok(xtx_inv) => {
                let beta = xtx_inv.dot(&xty);
                let predictions = x.dot(&beta);
                let residuals = &y.to_owned() - &predictions;
                let residual_std = residuals.std(0.0) + 1e-6;

                -0.5 * n as f64 * (2.0 * PI * residual_std * residual_std).ln()
                    - 0.5
                        * residuals
                            .iter()
                            .map(|&r| (r / residual_std).powi(2))
                            .sum::<f64>()
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

/// Score a parent set for a target variable
fn score_parent_set(data: &Array2<f64>, target: usize, parents: &[usize], lambda: f64) -> f64 {
    let log_lik = compute_log_likelihood(data, target, parents);
    let log_prior = -lambda * parents.len() as f64;
    log_lik + log_prior
}

/// Beam search to find best parent set for a single variable
fn beam_search_for_variable(
    data: &Array2<f64>,
    target: usize,
    n_vars: usize,
    beam_size: usize,
    max_rounds: usize,
    lambda: f64,
) -> (Vec<usize>, f64) {
    let predictors: Vec<usize> = (0..n_vars).filter(|&i| i != target).collect();

    // Initialize with single-variable theories + empty theory
    let mut theories: Vec<Vec<usize>> = predictors.iter().map(|&v| vec![v]).collect();
    theories.push(vec![]); // Empty theory

    // Score initial theories
    let mut scored: Vec<(Vec<usize>, f64)> = theories
        .into_iter()
        .map(|t| {
            let score = score_parent_set(data, target, &t, lambda);
            (t, score)
        })
        .collect();

    // Keep top k
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut current: Vec<(Vec<usize>, f64)> = scored.into_iter().take(beam_size).collect();
    let mut best_score = current[0].1;

    for _round in 0..max_rounds {
        let mut candidates = Vec::new();

        for (theory, old_score) in &current {
            // Keep original
            candidates.push((theory.clone(), *old_score));

            // Try adding each predictor
            for &v in &predictors {
                if !theory.contains(&v) {
                    let mut new_theory = theory.clone();
                    new_theory.push(v);
                    new_theory.sort();
                    let new_score = score_parent_set(data, target, &new_theory, lambda);
                    candidates.push((new_theory, new_score));
                }
            }
        }

        // Remove duplicates
        let mut seen = HashSet::new();
        let mut unique_candidates = Vec::new();
        for (theory, score) in candidates {
            let key = theory.clone();
            if seen.insert(key) {
                unique_candidates.push((theory, score));
            }
        }

        // Keep top k
        unique_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        current = unique_candidates.into_iter().take(beam_size).collect();

        let new_best = current[0].1;

        // Early stopping
        if new_best <= best_score + 0.1 {
            break;
        }

        best_score = new_best;
    }

    (current[0].0.clone(), current[0].1)
}

/// Beam search structure learning for all variables
fn beam_search_structure(
    data: &Array2<f64>,
    n_vars: usize,
    beam_size: usize,
    max_rounds: usize,
    lambda: f64,
) -> (HashMap<usize, Vec<usize>>, f64) {
    let mut structure = HashMap::new();
    let mut total_score = 0.0;

    println!("Running beam search for each variable (beam_size={}, max_rounds={})...", beam_size, max_rounds);
    println!();

    for var in 0..n_vars {
        let (parents, score) = beam_search_for_variable(data, var, n_vars, beam_size, max_rounds, lambda);

        if parents.is_empty() {
            println!("  var_{}: (no parents) | score: {:.2}", var, score);
        } else {
            println!("  var_{}: ← {:?} | score: {:.2}", var, parents, score);
        }

        structure.insert(var, parents);
        total_score += score;
    }

    println!();
    println!("Total structure score: {:.2}", total_score);

    (structure, total_score)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: secom_benchmark <n_features>");
        eprintln!("Example: secom_benchmark 50");
        std::process::exit(1);
    }

    let n_features: usize = args[1].parse().expect("Invalid n_features");
    if ![50, 75, 100].contains(&n_features) {
        eprintln!("Error: n_features must be 50, 75, or 100");
        std::process::exit(1);
    }

    println!("========================================");
    println!("SECOM {}-Variable Benchmark (Rust)", n_features);
    println!("========================================\n");

    // Load data
    println!("Loading SECOM {}-variable subset...", n_features);
    let (data, n_rows, n_cols) = load_secom_data(n_features);
    println!("  Shape: {} rows × {} columns\n", n_rows, n_cols);

    // Run beam search structure learning
    println!("Starting beam search structure learning...");
    println!("Beam size: 10, Max rounds: 5, Lambda: 2.0\n");

    let search_start = Instant::now();
    let (best_structure, best_score) = beam_search_structure(&data, n_cols, 10, 5, 2.0);
    let search_time = search_start.elapsed();

    println!("\n----------------------------------------");
    println!("RESULTS");
    println!("----------------------------------------");
    println!("Best score: {:.2}", best_score);
    println!("Elapsed time: {:.2}s\n", search_time.as_secs_f64());

    println!("Best structure:");
    for var in 0..n_cols {
        let parents = &best_structure[&var];
        if !parents.is_empty() {
            println!("  var_{} ← {:?}", var, parents);
        }
    }

    // Save results
    let home = env::var("HOME").unwrap();
    let results_dir = format!(
        "{}/Documents/research/experiments/001-baseline-scalability/results",
        home
    );
    std::fs::create_dir_all(&results_dir).expect("Failed to create results directory");

    let results_path = format!("{}/rust_secom_{}.txt", results_dir, n_features);

    let mut file = File::create(&results_path).expect("Failed to create results file");
    writeln!(file, "SECOM {}-Variable Benchmark (Rust)", n_features).unwrap();
    writeln!(file, "========================================\n").unwrap();
    writeln!(file, "Variables: {}", n_cols).unwrap();
    writeln!(file, "Samples: {}", n_rows).unwrap();
    writeln!(file, "Best score: {:.2}", best_score).unwrap();
    writeln!(file, "Elapsed time: {:.2}s\n", search_time.as_secs_f64()).unwrap();
    writeln!(file, "Best structure:").unwrap();
    for var in 0..n_cols {
        let parents = &best_structure[&var];
        if !parents.is_empty() {
            writeln!(file, "  var_{} ← {:?}", var, parents).unwrap();
        }
    }

    println!("\nResults saved to: {}", results_path);
}
