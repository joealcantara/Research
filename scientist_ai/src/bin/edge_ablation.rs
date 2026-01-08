use polars::prelude::*;
use scientist_ai::inference::{ScoringMethod, score_structure};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load Iris data
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("data/iris_numeric.csv".into()))?
        .finish()?;

    println!("=== Edge Ablation Analysis ===\n");
    println!("Dataset: Iris (5 variables, 150 samples)");
    println!("Scoring: Edge-based (λ=2.0)\n");

    // Best structure from 10-run experiment
    let mut full_structure: HashMap<String, Vec<String>> = HashMap::new();
    full_structure.insert("SepalLength".to_string(), vec!["PetalLength".to_string(), "SepalWidth".to_string()]);
    full_structure.insert("Species".to_string(), vec!["SepalWidth".to_string()]);
    full_structure.insert("PetalWidth".to_string(), vec!["Species".to_string()]);
    full_structure.insert("SepalWidth".to_string(), vec![]);
    full_structure.insert("PetalLength".to_string(), vec!["PetalWidth".to_string()]);

    let method = ScoringMethod::EdgeBased(2.0);
    let full_score = score_structure(&df, &full_structure, &method)?;

    println!("Full structure (5 edges): {:.3}\n", full_score);

    // Define edges to ablate
    let edges = vec![
        ("SepalLength", "PetalLength"),
        ("SepalLength", "SepalWidth"),
        ("Species", "SepalWidth"),
        ("PetalWidth", "Species"),
        ("PetalLength", "PetalWidth"),
    ];

    println!("Edge ablation (remove one edge at a time):\n");
    println!("{:<30} | {:>10} | {:>10}", "Edge Removed", "Score", "Δ Score");
    println!("{}", "-".repeat(55));

    for (child, parent) in &edges {
        // Create structure with this edge removed
        let mut ablated = full_structure.clone();
        if let Some(parents) = ablated.get_mut(&child.to_string()) {
            parents.retain(|p| p != parent);
        }

        let score = score_structure(&df, &ablated, &method)?;
        let delta = score - full_score;

        println!("{:<30} | {:>10.3} | {:>10.3}",
                 format!("{} ← {}", child, parent), score, delta);
    }

    println!("\nNote: More negative Δ = larger score drop = more important edge");

    Ok(())
}
