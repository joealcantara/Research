use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;

// Import modules from the library
use scientist_ai::{inference, search};
use scientist_ai::inference::{ScoringConfig, ScoringMethod};

/// Helper enum to deserialize either a single value or array
#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> From<OneOrMany<T>> for Vec<T> {
    fn from(value: OneOrMany<T>) -> Self {
        match value {
            OneOrMany::One(item) => vec![item],
            OneOrMany::Many(items) => items,
        }
    }
}

#[derive(Deserialize)]
struct Experiment {
    input: String,
    output: String,
    beam_size: Option<u8>,
    max_rounds: Option<u8>,

    // Optional: select specific columns (for large datasets like Numerai)
    #[serde(default)]
    columns: Option<Vec<String>>,

    // Optional: filter to specific era (for Numerai temporal data)
    #[serde(default)]
    era: Option<String>,

    // Legacy: single lambda parameter (deprecated in favor of scoring)
    lambda: Option<f64>,

    // New: flexible scoring configuration
    // Can be a single method or array of methods
    #[serde(default)]
    scoring: Option<OneOrMany<ScoringConfig>>,
}

/// Output structure for JSON results
#[derive(Serialize)]
struct ExperimentResults {
    config: ConfigInfo,
    data: DataInfo,
    methods: Vec<MethodResult>,
}

#[derive(Serialize)]
struct ConfigInfo {
    input: String,
    beam_size: usize,
    max_rounds: usize,
}

#[derive(Serialize)]
struct DataInfo {
    rows: usize,
    columns: usize,
    variables: Vec<String>,
}

#[derive(Serialize)]
struct MethodResult {
    method: String,
    best_structure: HashMap<String, Vec<String>>,
    edges: usize,
    score: f64,
}

fn main() {
    // Configure rayon thread pool with larger stack size for large-scale problems
    // Default 2MB per thread is insufficient for 596-variable problems
    rayon::ThreadPoolBuilder::new()
        .stack_size(16 * 1024 * 1024)  // 16MB per thread
        .build_global()
        .expect("Failed to configure rayon thread pool");

    println!("=== Scientist AI: Structure Learning ===\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <config.toml>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  cargo run experiments/example1_single_method.toml");
        std::process::exit(1);
    }
    let config_path = &args[1];

    // Load configuration
    let contents = fs::read_to_string(config_path)
        .expect("Failed to read config file");

    let config: Experiment = toml::from_str(&contents)
        .expect("Failed to parse TOML");

    let beam_size = config.beam_size.unwrap_or(10) as usize;
    let max_rounds = config.max_rounds.unwrap_or(3) as usize;

    // Parse scoring methods (with fallback to legacy lambda)
    let scoring_methods: Vec<ScoringMethod> = if let Some(scoring_configs) = config.scoring {
        let configs: Vec<ScoringConfig> = scoring_configs.into();
        configs.into_iter().map(|c| c.into()).collect()
    } else if let Some(lambda) = config.lambda {
        // Legacy: use lambda parameter
        vec![ScoringMethod::EdgeBased(lambda)]
    } else {
        // Default: EdgeBased with lambda=2.0
        vec![ScoringMethod::EdgeBased(2.0)]
    };

    // Print configuration
    println!("Configuration:");
    println!("  Input:      {}", config.input);
    println!("  Output:     {}", config.output);
    println!("  Beam size:  {}", beam_size);
    println!("  Max rounds: {}", max_rounds);
    println!("\nScoring methods ({}):", scoring_methods.len());
    for (i, method) in scoring_methods.iter().enumerate() {
        match method {
            ScoringMethod::EdgeBased(lambda) => {
                println!("  {}: Edge-based (λ={})", i + 1, lambda);
            }
            ScoringMethod::BIC => {
                println!("  {}: BIC", i + 1);
            }
            ScoringMethod::BDeu(alpha) => {
                println!("  {}: BDeu (α={})", i + 1, alpha);
            }
        }
    }

    // Load data (CSV or Parquet)
    println!("\n--- Loading Data ---");
    use polars::prelude::*;
    use std::path::Path;

    let input_path = Path::new(&config.input);
    let extension = input_path.extension()
        .and_then(|s| s.to_str())
        .unwrap_or("csv");

    let mut df = match extension {
        "parquet" => {
            println!("  Format: Parquet");
            LazyFrame::scan_parquet(&config.input, Default::default())
                .expect("Failed to scan parquet file")
                .collect()
                .expect("Failed to load parquet file")
        }
        "csv" => {
            println!("  Format: CSV");
            CsvReadOptions::default()
                .with_has_header(true)
                .try_into_reader_with_file_path(Some(config.input.clone().into()))
                .expect("Failed to create CSV reader")
                .finish()
                .expect("Failed to read CSV")
        }
        _ => {
            panic!("Unsupported file format: {}", extension);
        }
    };

    // Filter to specific era if requested
    if let Some(ref era_value) = config.era {
        println!("  Filtering to era: {}", era_value);
        df = df.lazy()
            .filter(col("era").eq(lit(era_value.as_str())))
            .collect()
            .expect("Failed to filter by era");
        println!("  Rows after era filter: {}", df.height());
    }

    // Select specific columns if requested
    if let Some(ref columns) = config.columns {
        println!("  Selecting {} columns from dataset", columns.len());
        df = df.select(columns)
            .expect("Failed to select columns");
    }

    // Cast all columns to Float64 (required for scoring)
    let cast_cols: Vec<Expr> = df.get_column_names()
        .iter()
        .map(|name| col(name.to_string()).cast(DataType::Float64))
        .collect();
    df = df.lazy()
        .select(&cast_cols)
        .collect()
        .expect("Failed to cast columns to Float64");

    println!("  Rows: {}", df.height());
    println!("  Columns: {}", df.width());
    println!("  Variables: {:?}", df.get_column_names());

    // Extract variable names
    let variables: Vec<String> = df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Prepare results collection
    let mut method_results = Vec::new();

    // Run structure learning for each scoring method
    for (i, method) in scoring_methods.iter().enumerate() {
        println!("\n=== Method {}/{}: {:?} ===", i + 1, scoring_methods.len(), method);

        let method_name = match method {
            ScoringMethod::EdgeBased(lambda) => format!("Edge-based (λ={})", lambda),
            ScoringMethod::BIC => "BIC".to_string(),
            ScoringMethod::BDeu(alpha) => format!("BDeu (α={})", alpha),
        };

        println!("\nRunning stochastic beam search...");

        let best_structure = search::stochastic_beam_search(
            &df,
            &variables,
            beam_size,
            max_rounds,
            method,
            1.0,  // temperature
            true  // verbose
        ).expect("Search failed");

        // Count edges in best structure
        let edges: usize = best_structure.values().map(|parents| parents.len()).sum();

        println!("\n--- Results for {} ---", method_name);
        println!("Best structure found ({} edges):", edges);

        // Print structure in readable format
        let mut sorted_vars: Vec<_> = best_structure.keys().collect();
        sorted_vars.sort();

        for var in sorted_vars {
            let parents = &best_structure[var];
            if parents.is_empty() {
                println!("  {} (no parents)", var);
            } else {
                println!("  {} ← {}", var, parents.join(", "));
            }
        }

        // Compute score for best structure
        let score = inference::score_structure(&df, &best_structure, method)
            .expect("Failed to score structure");
        println!("\nFinal score: {:.3}", score);

        // Store results
        method_results.push(MethodResult {
            method: method_name,
            best_structure: best_structure.clone(),
            edges,
            score,
        });
    }

    // Write JSON output
    println!("\n=== Writing Results ===");

    let results = ExperimentResults {
        config: ConfigInfo {
            input: config.input.clone(),
            beam_size,
            max_rounds,
        },
        data: DataInfo {
            rows: df.height(),
            columns: df.width(),
            variables: variables.clone(),
        },
        methods: method_results,
    };

    // Create output directory if needed
    if let Some(parent) = std::path::Path::new(&config.output).parent() {
        fs::create_dir_all(parent).expect("Failed to create output directory");
    }

    // Write JSON file
    let json = serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    fs::write(&config.output, json).expect("Failed to write output file");

    println!("Results written to: {}", config.output);
    println!("\n=== Complete ===");
}
