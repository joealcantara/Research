// Debug version to check specific structures against Python/Julia
use itertools::Itertools;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

const PI: f64 = std::f64::consts::PI;

type Structure = HashMap<usize, Vec<usize>>;

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
    let flat_data: Vec<f64> = data.into_iter().flatten().collect();

    (Array2::from_shape_vec((n_rows, 5), flat_data).unwrap(), n_rows)
}

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

pub fn main() {
    println!("DEBUG: Checking DAG enumeration");
    println!("================================");

    let (data, n_rows) = load_iris_data();
    println!("Loaded {} rows", n_rows);
    println!("First row: {:?}", data.row(0));
    println!("Last row: {:?}", data.row(n_rows - 1));
    println!();

    // Generate DAGs
    let all_dags = generate_all_dags(5);
    println!("Generated {} DAG structures", all_dags.len());
    println!();

    // Check specific structures that Python/Julia found as top:
    // Python/Julia top structure (28708/28709):
    // sepal_length: ← sepal_width, species
    // sepal_width: (no parents)
    // petal_length: ← sepal_length, sepal_width, petal_width, species
    // petal_width: ← sepal_length, species
    // species: ← sepal_width

    let mut target_structure = HashMap::new();
    target_structure.insert(0, vec![1, 4]); // sepal_length ← sepal_width, species
    target_structure.insert(1, vec![]);      // sepal_width ← (none)
    target_structure.insert(2, vec![0, 1, 3, 4]); // petal_length ← sepal_length, sepal_width, petal_width, species
    target_structure.insert(3, vec![0, 4]); // petal_width ← sepal_length, species
    target_structure.insert(4, vec![1]);     // species ← sepal_width

    // Find this structure in our list
    let mut found_idx = None;
    for (idx, structure) in all_dags.iter().enumerate() {
        if structure == &target_structure {
            found_idx = Some(idx);
            break;
        }
    }

    if let Some(idx) = found_idx {
        println!("✓ Found Python/Julia MAP structure at index: {}", idx);
        println!("  (Python uses 0-based: 28708, Julia uses 1-based: 28709)");
    } else {
        println!("✗ Python/Julia MAP structure NOT FOUND in Rust enumeration!");
        println!("  This suggests a problem with DAG generation");
    }
    println!();

    // Show what Rust found at index 28708
    if all_dags.len() > 28708 {
        println!("Rust structure at index 28708:");
        let var_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"];
        for i in 0..5 {
            let parents = &all_dags[28708][&i];
            if parents.is_empty() {
                println!("  {}: (no parents)", var_names[i]);
            } else {
                let parent_names: Vec<&str> = parents.iter().map(|&p| var_names[p]).collect();
                println!("  {}: ← {}", var_names[i], parent_names.join(", "));
            }
        }
    }
    println!();

    // Show Rust's top structure (25441)
    println!("Rust's top structure (index 25441):");
    let var_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"];
    for i in 0..5 {
        let parents = &all_dags[25441][&i];
        if parents.is_empty() {
            println!("  {}: (no parents)", var_names[i]);
        } else {
            let parent_names: Vec<&str> = parents.iter().map(|&p| var_names[p]).collect();
            println!("  {}: ← {}", var_names[i], parent_names.join(", "));
        }
    }
    println!();

    // Now compute scores for both structures
    println!("Computing scores for both structures:");
    println!("=====================================");

    use scientist_ai_rust::compute_log_likelihood_continuous;

    let lambda = 2.0;

    // Score structure 28708 (Python/Julia MAP)
    let mut log_lik_28708 = 0.0;
    let mut edges_28708 = 0;
    for var in 0..5 {
        let parents = &all_dags[28708][&var];
        let is_categorical = var == 4;
        let var_log_lik = compute_log_likelihood_continuous(&data, var, parents, is_categorical);
        log_lik_28708 += var_log_lik;
        edges_28708 += parents.len();
        println!("  Structure 28708, var {}: log_lik = {:.4}, parents = {:?}", var, var_log_lik, parents);
    }
    let log_prior_28708 = -lambda * edges_28708 as f64;
    let log_post_28708 = log_lik_28708 + log_prior_28708;
    println!("Structure 28708 total: log_lik={:.2}, edges={}, log_prior={:.2}, log_post={:.2}",
             log_lik_28708, edges_28708, log_prior_28708, log_post_28708);
    println!();

    // Score structure 25441 (Rust's top)
    let mut log_lik_25441 = 0.0;
    let mut edges_25441 = 0;
    for var in 0..5 {
        let parents = &all_dags[25441][&var];
        let is_categorical = var == 4;
        let var_log_lik = compute_log_likelihood_continuous(&data, var, parents, is_categorical);
        log_lik_25441 += var_log_lik;
        edges_25441 += parents.len();
        println!("  Structure 25441, var {}: log_lik = {:.4}, parents = {:?}", var, var_log_lik, parents);
    }
    let log_prior_25441 = -lambda * edges_25441 as f64;
    let log_post_25441 = log_lik_25441 + log_prior_25441;
    println!("Structure 25441 total: log_lik={:.2}, edges={}, log_prior={:.2}, log_post={:.2}",
             log_lik_25441, edges_25441, log_prior_25441, log_post_25441);
    println!();

    println!("Expected (from Python/Julia):");
    println!("  Structure 28708: log_post = -350.89");
    println!("  This structure should have the HIGHEST posterior");
}
