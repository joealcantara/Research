use polars::prelude::*;
use scientist_ai::inference::{compute_posteriors, ScoringMethod};
use scientist_ai::dag::is_dag;
use std::collections::HashMap;
use rand::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load data
    let data_path = "data/iris_numeric.csv";
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(data_path.into()))?
        .finish()?;

    let variables: Vec<String> = df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    println!("Testing if 1-edge and 2-edge optima are dominant\n");

    // Test 1-edge structures
    println!("=== 1-EDGE STRUCTURES ===");
    let mut one_edge_structures = HashMap::new();

    for child in &variables {
        for parent in &variables {
            if child == parent {
                continue;
            }

            let mut structure = HashMap::new();
            for var in &variables {
                if var == child {
                    structure.insert(var.clone(), vec![parent.clone()]);
                } else {
                    structure.insert(var.clone(), vec![]);
                }
            }

            let name = format!("{}→{}", parent, child);
            one_edge_structures.insert(name, structure);
        }
    }

    println!("Generated {} 1-edge structures", one_edge_structures.len());

    let method = ScoringMethod::BIC;
    let scores_1edge = compute_posteriors(&df, &one_edge_structures, &method)?;

    let mut score_vec: Vec<(String, f64)> = scores_1edge
        .iter()
        .map(|(name, s)| (name.clone(), s.log_posterior))
        .collect();
    score_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 10 1-edge structures:");
    for (i, (name, score)) in score_vec.iter().take(10).enumerate() {
        println!("  {}: {} = {:.3}", i+1, name, score);
    }

    let best_1edge = &score_vec[0];
    let second_1edge = &score_vec[1];
    let gap_1edge = best_1edge.1 - second_1edge.1;

    println!("\nBest: {} = {:.3}", best_1edge.0, best_1edge.1);
    println!("Gap to 2nd: {:.3}", gap_1edge);

    // Expected from beam search: -20934.220
    println!("Beam search found: -20934.220");
    println!("Match: {}", (best_1edge.1 - (-20934.220)).abs() < 0.01);

    // Test 2-edge structures (sample, not exhaustive)
    println!("\n=== 2-EDGE STRUCTURES (sample) ===");
    let mut two_edge_structures = HashMap::new();
    let mut rng = rand::thread_rng();

    // Sample 1000 random 2-edge structures
    for i in 0..1000 {
        let mut structure = HashMap::new();

        // Initialize
        for var in &variables {
            structure.insert(var.clone(), vec![]);
        }

        // Add 2 random edges
        let mut edges_added = 0;
        let mut attempts = 0;

        while edges_added < 2 && attempts < 100 {
            attempts += 1;

            let child = &variables[rng.gen_range(0..variables.len())];
            let parent = &variables[rng.gen_range(0..variables.len())];

            if child == parent {
                continue;
            }

            let mut new_structure = structure.clone();
            let mut parents = new_structure.get(child).cloned().unwrap_or_default();

            if parents.contains(parent) {
                continue;
            }

            parents.push(parent.clone());
            new_structure.insert(child.clone(), parents);

            if is_dag(&new_structure) {
                structure = new_structure;
                edges_added += 1;
            }
        }

        if edges_added == 2 {
            two_edge_structures.insert(format!("2edge_{}", i), structure);
        }
    }

    println!("Generated {} 2-edge structures", two_edge_structures.len());

    let scores_2edge = compute_posteriors(&df, &two_edge_structures, &method)?;

    let mut score_vec_2: Vec<(String, f64)> = scores_2edge
        .iter()
        .map(|(name, s)| (name.clone(), s.log_posterior))
        .collect();
    score_vec_2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 10 2-edge structures:");
    for (i, (name, score)) in score_vec_2.iter().take(10).enumerate() {
        println!("  {}: Score = {:.3}", i+1, score);
    }

    let best_2edge = &score_vec_2[0];
    let worst_2edge = &score_vec_2[score_vec_2.len() - 1];

    println!("\nBest 2-edge: {:.3}", best_2edge.1);
    println!("Worst 2-edge: {:.3}", worst_2edge.1);
    println!("Range: {:.3}", best_2edge.1 - worst_2edge.1);

    // Expected from beam search: -20926.151
    println!("\nBeam search found: -20926.151");
    println!("Our best: {:.3}", best_2edge.1);
    println!("Gap: {:.3}", best_2edge.1 - (-20926.151));

    Ok(())
}
