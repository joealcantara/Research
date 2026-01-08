/// Validation tests: Compare Rust implementation against Python/Julia on Iris dataset
use polars::prelude::*;
use scientist_ai::inference::{score_structure, ScoringMethod};
use std::collections::HashMap;

#[test]
fn test_score_julia_map_structure() {
    println!("\n=== Phase 1: Score Validation on Julia MAP Structure ===\n");

    // Load Iris dataset (numeric version: Species encoded as 1,2,3)
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("data/iris_numeric.csv".into()))
        .expect("Failed to create CSV reader")
        .finish()
        .expect("Failed to read iris_numeric.csv");

    println!("Loaded Iris dataset:");
    println!("  Rows: {}", df.height());
    println!("  Columns: {:?}", df.get_column_names());

    // Julia MAP structure (from IRIS_VALIDATION.md)
    // sepal_length: ← sepal_width, species
    // sepal_width: (no parents)
    // petal_length: ← sepal_length, sepal_width, petal_width, species
    // petal_width: ← sepal_length, species
    // species: ← sepal_width
    //
    // Total edges: 9
    // Julia posterior: 0.174838
    // Julia log-posterior: -350.89

    let mut map_structure = HashMap::new();
    map_structure.insert("SepalLength".to_string(), vec!["SepalWidth".to_string(), "Species".to_string()]);
    map_structure.insert("SepalWidth".to_string(), vec![]);
    map_structure.insert("PetalLength".to_string(), vec![
        "SepalLength".to_string(),
        "SepalWidth".to_string(),
        "PetalWidth".to_string(),
        "Species".to_string(),
    ]);
    map_structure.insert("PetalWidth".to_string(), vec!["SepalLength".to_string(), "Species".to_string()]);
    map_structure.insert("Species".to_string(), vec!["SepalWidth".to_string()]);

    let edges: usize = map_structure.values().map(|p| p.len()).sum();
    println!("\nJulia MAP Structure ({} edges):", edges);
    for (var, parents) in &map_structure {
        if parents.is_empty() {
            println!("  {} (no parents)", var);
        } else {
            println!("  {} ← {}", var, parents.join(", "));
        }
    }

    // Score with Edge-based (λ=2.0) - what Julia/Python used for continuous Iris
    println!("\n=== Scoring with Edge-based (λ=2.0) ===");
    let method = ScoringMethod::EdgeBased(2.0);
    let score = score_structure(&df, &map_structure, &method)
        .expect("Failed to score structure");

    println!("\nRust Results:");
    println!("  Edge-based score: {:.6}", score);

    println!("\nJulia Reference (from IRIS_VALIDATION.md):");
    println!("  Log-posterior: -350.89");
    println!("  Posterior: 0.174838");

    // Note: The score should be close to -350.89
    // Exact match depends on:
    // 1. Same scoring formula (Edge-based with λ=2.0)
    // 2. Same handling of continuous variables (Linear Gaussian)
    // 3. Same handling of mixed variables (Species is categorical)

    println!("\n=== Phase 1 Status ===");
    println!("Score computed: {:.6}", score);
    println!("Expected: ~-350.89");

    let diff = (score - (-350.89)).abs();
    println!("Difference: {:.6}", diff);

    if diff < 1.0 {
        println!("✓ CLOSE MATCH (within 1.0)");
    } else {
        println!("⚠ MISMATCH - Need to investigate:");
        println!("  - Check variable type detection (continuous vs categorical)");
        println!("  - Check discretization method");
        println!("  - Check BDeu formula");
    }
}

#[test]
fn test_score_with_all_methods() {
    println!("\n=== Scoring Julia MAP with All Three Methods ===\n");

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("data/iris.csv".into()))
        .expect("Failed to create CSV reader")
        .finish()
        .expect("Failed to read iris.csv");

    let mut map_structure = HashMap::new();
    map_structure.insert("SepalLength".to_string(), vec!["SepalWidth".to_string(), "Species".to_string()]);
    map_structure.insert("SepalWidth".to_string(), vec![]);
    map_structure.insert("PetalLength".to_string(), vec![
        "SepalLength".to_string(),
        "SepalWidth".to_string(),
        "PetalWidth".to_string(),
        "Species".to_string(),
    ]);
    map_structure.insert("PetalWidth".to_string(), vec!["SepalLength".to_string(), "Species".to_string()]);
    map_structure.insert("Species".to_string(), vec!["SepalWidth".to_string()]);

    println!("Testing all three scoring methods on Julia MAP structure:\n");

    let methods = vec![
        ("Edge-based (λ=2.0)", ScoringMethod::EdgeBased(2.0)),
        ("BIC", ScoringMethod::BIC),
        ("BDeu (α=1.0)", ScoringMethod::BDeu(1.0)),
    ];

    for (name, method) in methods {
        let score = score_structure(&df, &map_structure, &method)
            .expect("Failed to score");
        println!("{:20} Score: {:.6}", name, score);
    }
}
