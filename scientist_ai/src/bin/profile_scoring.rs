use polars::prelude::*;
use scientist_ai::search::generate_neighbors;
use scientist_ai::inference::{score_structure, ScoringMethod};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load Iris data
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("data/iris_numeric.csv".into()))?
        .finish()?;

    println!("Loaded Iris: {} rows, {} columns", df.height(), df.width());

    let variables: Vec<String> = df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Create empty structure
    let mut empty_structure = HashMap::new();
    for var in &variables {
        empty_structure.insert(var.clone(), vec![]);
    }

    println!("\n=== Timing Analysis ===\n");

    // Generate neighbors (21 structures)
    println!("Generating neighbors...");
    let start = Instant::now();
    let neighbors = generate_neighbors(&empty_structure, &variables);
    println!("Generated {} neighbors in {:?}", neighbors.len(), start.elapsed());

    // Score each structure
    let method = ScoringMethod::EdgeBased(2.0);
    println!("\nScoring {} structures with EdgeBased(2.0)...", neighbors.len());

    let start = Instant::now();
    for (i, structure) in neighbors.iter().enumerate() {
        let iter_start = Instant::now();
        let score = score_structure(&df, structure, &method)?;
        let elapsed = iter_start.elapsed();

        if i < 5 || elapsed.as_secs_f64() > 1.0 {
            let edges: usize = structure.values().map(|p| p.len()).sum();
            println!("  Structure {}: score={:.3}, edges={}, time={:?}",
                     i+1, score, edges, elapsed);
        }
    }
    let total = start.elapsed();

    println!("\nTotal time: {:?}", total);
    println!("Average per structure: {:?}", total / neighbors.len() as u32);
    println!("Structures per second: {:.2}", neighbors.len() as f64 / total.as_secs_f64());

    Ok(())
}
