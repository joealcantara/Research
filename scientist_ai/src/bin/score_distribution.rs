use polars::prelude::*;
use scientist_ai::inference::{compute_posteriors, ScoringMethod};
use scientist_ai::dag::is_dag;
use std::collections::HashMap;
use rand::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load data
    let data_path = "data/secom_6var.csv";
    println!("Loading data from: {}", data_path);
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(data_path.into()))?
        .finish()?;

    let variables: Vec<String> = df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    println!("Variables: {:?}", variables);
    println!("\nGenerating random DAG structures...");

    let mut rng = rand::thread_rng();
    let n_samples = 10000;
    let mut structures = HashMap::new();

    // Generate random DAG structures
    for i in 0..n_samples {
        let mut structure = HashMap::new();

        // Initialize all variables
        for var in &variables {
            structure.insert(var.clone(), vec![]);
        }

        // Randomly add edges (with low probability to avoid cycles)
        for child in &variables {
            for parent in &variables {
                if child == parent {
                    continue;
                }

                // 20% chance of edge
                if rng.gen_bool(0.2) {
                    let mut new_structure = structure.clone();
                    let mut parents = new_structure.get(child).cloned().unwrap_or_default();
                    parents.push(parent.clone());
                    new_structure.insert(child.clone(), parents);

                    // Check if still a DAG
                    if is_dag(&new_structure) {
                        structure = new_structure;
                    }
                }
            }
        }

        structures.insert(format!("random_{}", i), structure);
    }

    println!("Generated {} random DAG structures", structures.len());
    println!("\nScoring with BIC...");

    // Score all structures
    let method = ScoringMethod::BIC;
    let scores = compute_posteriors(&df, &structures, &method)?;

    // Collect scores
    let mut score_values: Vec<f64> = scores.values()
        .map(|s| s.log_posterior)
        .collect();
    score_values.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Statistics
    let best_score = score_values[0];
    let worst_score = score_values[score_values.len() - 1];
    let mean_score = score_values.iter().sum::<f64>() / score_values.len() as f64;
    let median_score = score_values[score_values.len() / 2];

    // Beam search optimum
    let beam_optimum = -20923.604;

    println!("\n=== SCORE DISTRIBUTION ===");
    println!("Random samples: {}", score_values.len());
    println!("Best random:    {:.3}", best_score);
    println!("Worst random:   {:.3}", worst_score);
    println!("Mean random:    {:.3}", mean_score);
    println!("Median random:  {:.3}", median_score);
    println!("\nBeam search optimum: {:.3}", beam_optimum);
    println!("Gap (beam - best random): {:.3}", beam_optimum - best_score);

    // Percentiles
    println!("\n=== PERCENTILES ===");
    for pct in [99, 95, 90, 75, 50, 25, 10].iter() {
        let idx = (score_values.len() * pct / 100).min(score_values.len() - 1);
        println!("{}th percentile: {:.3}", pct, score_values[idx]);
    }

    // Count how many are close to beam optimum
    let within_1 = score_values.iter().filter(|&&s| s >= beam_optimum - 1.0).count();
    let within_10 = score_values.iter().filter(|&&s| s >= beam_optimum - 10.0).count();
    let within_100 = score_values.iter().filter(|&&s| s >= beam_optimum - 100.0).count();

    println!("\n=== PROXIMITY TO BEAM OPTIMUM ===");
    println!("Within 1.0 of beam optimum:   {} ({:.2}%)", within_1, 100.0 * within_1 as f64 / score_values.len() as f64);
    println!("Within 10.0 of beam optimum:  {} ({:.2}%)", within_10, 100.0 * within_10 as f64 / score_values.len() as f64);
    println!("Within 100.0 of beam optimum: {} ({:.2}%)", within_100, 100.0 * within_100 as f64 / score_values.len() as f64);

    // Histogram
    println!("\n=== HISTOGRAM (10 bins) ===");
    let bin_size = (best_score - worst_score) / 10.0;
    for i in 0..10 {
        let bin_start = best_score - (i as f64 + 1.0) * bin_size;
        let bin_end = best_score - i as f64 * bin_size;
        let count = score_values.iter().filter(|&&s| s >= bin_start && s < bin_end).count();
        let bar = "#".repeat((count as f64 / score_values.len() as f64 * 50.0) as usize);
        println!("[{:8.1}, {:8.1}): {:5} {}", bin_start, bin_end, count, bar);
    }

    Ok(())
}
