use itertools::Itertools;
use ndarray::{s, Array1, Array2};
use std::collections::HashMap;
use std::time::Instant;

const PI: f64 = std::f64::consts::PI;

type Structure = HashMap<usize, Vec<usize>>;

/// Load Iris dataset from CSV
fn load_iris_data() -> (Array2<f64>, usize) {
    let data_path = std::env::var("HOME").unwrap() + "/Documents/projects/learning/scientist_ai_julia/data/iris.csv";

    let mut reader = csv::Reader::from_path(&data_path).expect("Failed to open iris.csv");
    let mut data = Vec::new();

    for result in reader.records() {
        let record = result.unwrap();
        let sepal_length: f64 = record[0].parse().unwrap();
        let sepal_width: f64 = record[1].parse().unwrap();
        let petal_length: f64 = record[2].parse().unwrap();
        let petal_width: f64 = record[3].parse().unwrap();
        let species_str = &record[4];
        let species: f64 = match species_str {
            "setosa" => 1.0,
            "versicolor" => 2.0,
            "virginica" => 3.0,
            _ => panic!("Unknown species"),
        };

        data.push(vec![sepal_length, sepal_width, petal_length, petal_width, species]);
    }

    let n_rows = data.len();
    let n_cols = 5;
    let flat_data: Vec<f64> = data.into_iter().flatten().collect();

    (Array2::from_shape_vec((n_rows, n_cols), flat_data).unwrap(), n_rows)
}

/// Check if edges form a DAG using DFS
fn is_dag(edges: &[(usize, usize)], n: usize) -> bool {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(i, j) in edges {
        adj[i].push(j);
    }

    let mut visited = vec![false; n];
    let mut rec_stack = vec![false; n];

    fn has_cycle(node: usize, adj: &[Vec<usize>], visited: &mut [bool], rec_stack: &mut [bool]) -> bool {
        visited[node] = true;
        rec_stack[node] = true;

        for &neighbor in &adj[node] {
            if !visited[neighbor] {
                if has_cycle(neighbor, adj, visited, rec_stack) {
                    return true;
                }
            } else if rec_stack[neighbor] {
                return true;
            }
        }

        rec_stack[node] = false;
        false
    }

    for i in 0..n {
        if !visited[i] && has_cycle(i, &adj, &mut visited, &mut rec_stack) {
            return false;
        }
    }
    true
}

/// Generate all possible DAG structures
fn generate_all_dags(n: usize) -> Vec<Structure> {
    let mut all_edges = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                all_edges.push((i, j));
            }
        }
    }

    let mut dags = Vec::new();

    for num_edges in 0..=all_edges.len() {
        for edge_subset in all_edges.iter().copied().combinations(num_edges) {
            if is_dag(&edge_subset, n) {
                let mut structure = HashMap::new();
                for i in 0..n {
                    let parents: Vec<usize> = edge_subset
                        .iter()
                        .filter_map(|&(j, k)| if k == i { Some(j) } else { None })
                        .collect();
                    structure.insert(i, parents);
                }
                dags.push(structure);
            }
        }
    }

    dags
}

/// Compute log-likelihood for continuous variable with linear Gaussian model
pub fn compute_log_likelihood_continuous(
    data: &Array2<f64>,
    var: usize,
    parents: &[usize],
    is_categorical: bool,
) -> f64 {
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

/// Compute posterior for all structures
fn compute_posteriors(data: &Array2<f64>, structures: &[Structure], lambda: f64) -> Vec<(f64, f64, usize)> {
    structures
        .iter()
        .enumerate()
        .map(|(idx, structure)| {
            let mut log_lik = 0.0;
            let mut edges = 0;

            for var in 0..5 {
                let parents = &structure[&var];
                let is_categorical = var == 4; // Species is categorical
                log_lik += compute_log_likelihood_continuous(data, var, parents, is_categorical);
                edges += parents.len();
            }

            let log_prior = -lambda * edges as f64;
            let log_posterior = log_lik + log_prior;

            (log_posterior, log_lik, idx)
        })
        .collect()
}

fn main() {
    println!("================================================================================");
    println!("Iris Dataset - Continuous Variables Benchmark (Rust)");
    println!("================================================================================");
    println!();

    // Load data
    println!("STEP 1: Load Iris dataset (continuous)");
    println!("--------------------------------------------------------------------------------");
    let (data, n_rows) = load_iris_data();
    println!("Dataset shape: ({}, 5)", n_rows);
    println!();

    // Generate all DAGs
    println!("STEP 2: Enumerate all possible DAG structures");
    println!("--------------------------------------------------------------------------------");
    println!("Generating all DAGs...");
    let start = Instant::now();
    let all_dags = generate_all_dags(5);
    let dag_time = start.elapsed();
    println!("Generated {} possible DAG structures in {:?}", all_dags.len(), dag_time);
    println!();

    // Compute posteriors
    println!("STEP 3: Compute posterior probabilities");
    println!("--------------------------------------------------------------------------------");
    println!("Running Bayesian inference with λ = 2.0 (complexity penalty)");
    println!("Using Linear Gaussian models for continuous variables");
    println!();

    let start = Instant::now();
    let mut results = compute_posteriors(&data, &all_dags, 2.0);
    let inference_time = start.elapsed();

    // Sort by posterior (descending)
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Normalize posteriors
    let max_log_post = results[0].0;
    let log_posts_shifted: Vec<f64> = results.iter().map(|(lp, _, _)| lp - max_log_post).collect();
    let posts: Vec<f64> = log_posts_shifted.iter().map(|lp| lp.exp()).collect();
    let sum_posts: f64 = posts.iter().sum();
    let normalized_posts: Vec<f64> = posts.iter().map(|p| p / sum_posts).collect();

    println!("Computed posteriors for all {} structures in {:?}", results.len(), inference_time);
    println!();

    // Show results
    println!("STEP 4: Analyze results");
    println!("--------------------------------------------------------------------------------");
    println!();
    println!("Top 20 structures by posterior probability:");
    println!();

    for (i, ((log_post, log_lik, idx), &post)) in results.iter().zip(&normalized_posts).take(20).enumerate() {
        let edges: usize = all_dags[*idx].values().map(|p| p.len()).sum();
        println!(
            "{:3}. Structure {:5}: posterior = {:.6}   ({:.2}%)  |  edges = {}  |  log-post = {:.2}",
            i + 1,
            idx,
            post,
            post * 100.0,
            edges,
            log_post
        );
    }
    println!();

    // Posterior concentration
    let top_1 = normalized_posts[0];
    let top_10: f64 = normalized_posts.iter().take(10).sum();
    let top_100: f64 = normalized_posts.iter().take(100).sum();

    println!("Posterior concentration:");
    println!("  Top 1:    {:.6} ({:.2}%)", top_1, top_1 * 100.0);
    println!("  Top 10:   {:.6} ({:.2}%)", top_10, top_10 * 100.0);
    println!("  Top 100:  {:.6} ({:.2}%)", top_100, top_100 * 100.0);
    println!();

    // MAP structure
    println!("================================================================================");
    println!("MAP (Maximum A Posteriori) Structure");
    println!("================================================================================");
    println!();

    let (_, _, map_idx) = results[0];
    let map_structure = &all_dags[map_idx];
    let var_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"];

    for i in 0..5 {
        let parents = &map_structure[&i];
        if parents.is_empty() {
            println!("{}: (no parents - root node)", var_names[i]);
        } else {
            let parent_names: Vec<&str> = parents.iter().map(|&p| var_names[p]).collect();
            println!("{}: ← {}", var_names[i], parent_names.join(", "));
        }
    }

    let total_edges: usize = map_structure.values().map(|p| p.len()).sum();
    println!("\nTotal edges: {}", total_edges);
    println!("Posterior probability: {:.6}", top_1);
    println!();

    // Total time
    let total_time = dag_time + inference_time;
    println!("================================================================================");
    println!("TIMING SUMMARY");
    println!("================================================================================");
    println!("DAG generation:      {:?}", dag_time);
    println!("Posterior inference: {:?}", inference_time);
    println!("Total time:          {:?}", total_time);
    println!();

    println!("================================================================================");
    println!("BENCHMARK COMPLETE");
    println!("================================================================================");
}
