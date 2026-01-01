use itertools::Itertools;
use ndarray::Array2;
use std::collections::HashMap;
use std::time::Instant;

type Structure = HashMap<usize, Vec<usize>>;

fn load_wine_data() -> (Array2<f64>, usize, Vec<String>) {
    let data_path = std::env::var("HOME").unwrap() + "/Documents/projects/learning/scientist_ai_julia/data/wine.csv";
    let mut reader = csv::Reader::from_path(&data_path).expect("Failed to open wine.csv");

    let headers: Vec<String> = reader.headers().unwrap().iter().map(|s| s.to_string()).collect();
    let mut data = Vec::new();

    for result in reader.records() {
        let record = result.unwrap();
        let row: Vec<f64> = record.iter().map(|s| s.parse().unwrap()).collect();
        data.extend(row);
    }

    let n_cols = headers.len();
    let n_rows = data.len() / n_cols;

    (Array2::from_shape_vec((n_rows, n_cols), data).unwrap(), n_rows, headers)
}

fn get_variable_type(data: &Array2<f64>, var: usize) -> bool {
    // Simple heuristic: categorical if <= 10 unique values
    let col = data.column(var);
    let mut unique_vals: Vec<f64> = col.iter().copied().collect();
    unique_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_vals.dedup();
    unique_vals.len() <= 10
}

fn custom_beam_search<F>(
    n_vars: usize,
    target: usize,
    score_fn: F,
    k: usize,
    max_rounds: usize,
) -> Vec<(Vec<usize>, f64)>
where
    F: Fn(&[usize]) -> f64,
{
    let predictors: Vec<usize> = (0..n_vars).filter(|&v| v != target).collect();

    // Initialize with single-variable theories + empty
    let mut theories: Vec<Vec<usize>> = predictors.iter().map(|&v| vec![v]).collect();
    theories.push(vec![]);

    let mut scored: Vec<(Vec<usize>, f64)> = theories.iter()
        .map(|t| (t.clone(), score_fn(t)))
        .collect();

    // Sort and keep top k
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut current = scored.into_iter().take(k).collect::<Vec<_>>();
    let mut best_score = current[0].1;

    for _round in 0..max_rounds {
        let mut candidates = Vec::new();

        for (theory, old_score) in &current {
            candidates.push((theory.clone(), *old_score));

            for &v in &predictors {
                if !theory.contains(&v) {
                    let mut new_theory = theory.clone();
                    new_theory.push(v);
                    let new_score = score_fn(&new_theory);
                    candidates.push((new_theory, new_score));
                }
            }
        }

        // Remove duplicates
        let mut seen = std::collections::HashSet::new();
        let mut unique_candidates = Vec::new();
        for (theory, score) in candidates {
            let mut sorted_theory = theory.clone();
            sorted_theory.sort();
            let key = format!("{:?}", sorted_theory);
            if seen.insert(key) {
                unique_candidates.push((theory, score));
            }
        }

        // Sort and keep top k
        unique_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        current = unique_candidates.into_iter().take(k).collect();

        let new_best = current[0].1;
        if new_best <= best_score + 0.1 {
            break;
        }
        best_score = new_best;
    }

    current
}

fn main() {
    use scientist_ai_rust::compute_log_likelihood_continuous;

    println!("================================================================================");
    println!("Wine Dataset: Beam Search Structure Learning (Rust)");
    println!("================================================================================");
    println!();

    // Load data
    println!("Loading Wine dataset...");
    let (data, n_rows, headers) = load_wine_data();
    let n_vars = headers.len();

    println!("Dataset: {} samples, {} variables", n_rows, n_vars);
    println!();

    println!("Variables:");
    for (i, header) in headers.iter().enumerate() {
        let is_categorical = get_variable_type(&data, i);
        if is_categorical {
            println!("  {:2}. {:<30} - categorical", i + 1, header);
        } else {
            let col = data.column(i);
            let mean = col.mean().unwrap();
            let std = col.std(0.0);
            println!("  {:2}. {:<30} - continuous (mean={:.2}, std={:.2})", i + 1, header, mean, std);
        }
    }
    println!();

    // Approach 1: Pure Linear Gaussian
    println!("================================================================================");
    println!("Approach 1: Pure Linear Gaussian");
    println!("================================================================================");
    println!();

    println!("Running beam search for each variable (beam_size=10, max_rounds=5)...");
    println!();

    let lambda = 2.0;
    let start_total = Instant::now();

    let mut structure = HashMap::new();
    let mut total_score = 0.0;

    for var in 0..n_vars {
        let is_categorical = get_variable_type(&data, var);

        let score_fn = |parents: &[usize]| {
            compute_log_likelihood_continuous(&data, var, parents, is_categorical)
                - lambda * parents.len() as f64
        };

        let results = custom_beam_search(n_vars, var, score_fn, 10, 5);

        if !results.is_empty() {
            let (best_parents, best_score) = &results[0];
            structure.insert(var, best_parents.clone());
            total_score += best_score;

            if best_parents.is_empty() {
                println!("  {}: (no parents) | score: {:.2} | theory: linear", headers[var], best_score);
            } else {
                let parent_names: Vec<&str> = best_parents.iter().map(|&p| headers[p].as_str()).collect();
                println!("  {}: ← {} | score: {:.2} | theory: linear",
                         headers[var], parent_names.join(", "), best_score);
            }
        } else {
            structure.insert(var, vec![]);
            println!("  {}: (no parents - no results)", headers[var]);
        }
    }

    let elapsed = start_total.elapsed();

    println!();
    println!("Total structure score: {:.2}", total_score);
    println!();

    println!("================================================================================");
    println!("TIMING SUMMARY");
    println!("================================================================================");
    println!("Total time: {:?}", elapsed);
    println!();

    println!("================================================================================");
    println!("COMPARISON");
    println!("================================================================================");
    println!();
    println!("Python:  6.5s");
    println!("Julia:   2.0s");
    println!("Rust:    {:?} ({:.1}x faster than Python, {:.1}x faster than Julia)",
             elapsed,
             6.5 / elapsed.as_secs_f64(),
             2.0 / elapsed.as_secs_f64());
    println!();

    println!("================================================================================");
    println!("BENCHMARK COMPLETE");
    println!("================================================================================");
}
