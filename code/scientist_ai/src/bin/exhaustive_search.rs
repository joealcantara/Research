use polars::prelude::*;
use scientist_ai::search::enumerate_all_dags;
use scientist_ai::inference::{compute_posteriors, ScoringMethod};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load data
    let data_path = "data/chain_3var.csv";
    println!("Loading data from: {}", data_path);
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(data_path.into()))?
        .finish()?;

    let n_samples = df.height();
    println!("Loaded {} samples", n_samples);

    // Get variables
    let variables: Vec<String> = df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    println!("Variables: {:?}", variables);

    // Enumerate all DAGs
    println!("\nEnumerating all possible DAG structures...");
    let all_dags = enumerate_all_dags(&variables);
    println!("Found {} DAGs", all_dags.len());

    // Create named structures for scoring
    let mut named_structures = HashMap::new();
    for (i, dag) in all_dags.iter().enumerate() {
        // Create a canonical name based on edges
        let mut edges = Vec::new();
        for (child, parents) in dag {
            for parent in parents {
                edges.push(format!("{}->{}", parent, child));
            }
        }
        edges.sort();
        let name = if edges.is_empty() {
            "empty".to_string()
        } else {
            edges.join(",")
        };

        named_structures.insert(name, dag.clone());
    }

    // Score all structures with BDeu
    println!("\nScoring all structures with BDeu...");
    let method = ScoringMethod::BDeu(1.0);
    let scores = compute_posteriors(&df, &named_structures, &method)?;

    // Sort by posterior probability
    let mut scored_structures: Vec<_> = scores.iter().collect();
    scored_structures.sort_by(|a, b| {
        b.1.posterior.partial_cmp(&a.1.posterior).unwrap()
    });

    // Print top 10
    println!("\n=== Top 10 Structures ===");
    for (i, (name, score)) in scored_structures.iter().take(10).enumerate() {
        println!("{:2}. {} (edges: {}, posterior: {:.6}, log_post: {:.3})",
            i + 1,
            name,
            score.edges,
            score.posterior,
            score.log_posterior
        );
    }

    // Save full results to JSON
    let output_path = "results/exhaustive_rust.json";
    println!("\nSaving results to: {}", output_path);

    let mut results = serde_json::json!({
        "n_samples": n_samples,
        "n_variables": variables.len(),
        "n_structures": scored_structures.len(),
        "scoring_method": "BDeu",
        "structures": []
    });

    let structures_array = results["structures"].as_array_mut().unwrap();
    for (name, score) in &scored_structures {
        structures_array.push(serde_json::json!({
            "name": name,
            "edges": score.edges,
            "log_likelihood": score.log_likelihood,
            "log_prior": score.log_prior,
            "log_posterior": score.log_posterior,
            "posterior": score.posterior,
        }));
    }

    let mut file = File::create(output_path)?;
    file.write_all(serde_json::to_string_pretty(&results)?.as_bytes())?;

    println!("\n=== Summary Statistics ===");
    println!("Total probability mass: {:.6}",
        scored_structures.iter().map(|(_, s)| s.posterior).sum::<f64>());
    println!("Entropy: {:.3} nats",
        -scored_structures.iter()
            .map(|(_, s)| if s.posterior > 0.0 { s.posterior * s.posterior.ln() } else { 0.0 })
            .sum::<f64>());

    println!("\nDone!");
    Ok(())
}
