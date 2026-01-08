/// Debug: Detailed scoring breakdown
use polars::prelude::*;
use scientist_ai::inference::{get_variable_type, estimate_binary_parameters, estimate_categorical_parameters, estimate_continuous_parameters};
use std::collections::HashMap;

#[test]
fn test_debug_iris_scoring() {
    println!("\n=== Debug: Iris Scoring Breakdown ===\n");

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("data/iris.csv".into()))
        .expect("Failed to read CSV")
        .finish()
        .expect("Failed to load CSV");

    // Julia MAP structure
    let mut structure = HashMap::new();
    structure.insert("SepalLength".to_string(), vec!["SepalWidth".to_string(), "Species".to_string()]);
    structure.insert("SepalWidth".to_string(), vec![]);
    structure.insert("PetalLength".to_string(), vec![
        "SepalLength".to_string(),
        "SepalWidth".to_string(),
        "PetalWidth".to_string(),
        "Species".to_string(),
    ]);
    structure.insert("PetalWidth".to_string(), vec!["SepalLength".to_string(), "Species".to_string()]);
    structure.insert("Species".to_string(), vec!["SepalWidth".to_string()]);

    println!("Structure:");
    for (var, parents) in &structure {
        let var_type = get_variable_type(&df, var).unwrap();
        print!("  {:15} {:?} ", var, var_type);
        if parents.is_empty() {
            println!("(no parents)");
        } else {
            let parent_types: Vec<String> = parents
                .iter()
                .map(|p| format!("{:?}", get_variable_type(&df, p).unwrap()))
                .collect();
            println!("← {} ({})", parents.join(", "), parent_types.join(", "));
        }
    }

    println!("\n=== Attempting Parameter Estimation ===\n");

    // Try estimating parameters for each variable type
    for (var, parents) in &structure {
        let var_type = get_variable_type(&df, var).unwrap();
        println!("Variable: {} ({:?})", var, var_type);
        println!("  Parents: {:?}", parents);

        match var_type {
            scientist_ai::inference::VariableType::Binary => {
                println!("  -> Attempting binary parameter estimation");
                // Would call estimate_binary_parameters
            }
            scientist_ai::inference::VariableType::Categorical => {
                println!("  -> Attempting categorical parameter estimation");
                // Would call estimate_categorical_parameters
            }
            scientist_ai::inference::VariableType::Continuous => {
                println!("  -> Attempting continuous parameter estimation");
                // Check parent types
                for parent in parents {
                    let parent_type = get_variable_type(&df, parent).unwrap();
                    println!("     Parent {} is {:?}", parent, parent_type);
                }
            }
        }
        println!();
    }

    println!("\nKey Question: How to handle mixed networks?");
    println!("- SepalLength (Continuous) ← SepalWidth (Continuous), Species (Categorical)");
    println!("- PetalLength (Continuous) ← 3 Continuous + 1 Categorical");
    println!("- PetalWidth (Continuous) ← SepalLength (Continuous), Species (Categorical)");
    println!("- Species (Categorical) ← SepalWidth (Continuous)");
    println!("\nMixed networks need special handling!");
    println!("Current code: estimate_categorical_parameters() for Species with continuous parent?");
}
