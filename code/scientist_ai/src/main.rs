use serde::Deserialize;
use std::env;
use std::fs;

// Import modules from the library
use scientist_ai::{dag, inference, search};
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

    // Legacy: single lambda parameter (deprecated in favor of scoring)
    lambda: Option<f64>,

    // New: flexible scoring configuration
    // Can be a single method or array of methods
    #[serde(default)]
    scoring: Option<OneOrMany<ScoringConfig>>,
}

fn main() {
    println!("Scientist AI");
    println!("--Check connectivity--");
    inference::hello();

    let args: Vec<String> = env::args().collect();
    let config_path = &args[1];

    let contents = fs::read_to_string(config_path)
        .expect("Failed to read config file");

    let config: Experiment = toml::from_str(&contents)
        .expect("Failed to parse TOML");

    let beam_size = config.beam_size.unwrap_or(10);
    let max_rounds = config.max_rounds.unwrap_or(3);

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

    println!("Input: {}", config.input);
    println!("Output: {}", config.output);
    println!("Beam size: {}", beam_size);
    println!("Max rounds: {}", max_rounds);
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

    let mut dag = dag::DAG::new();

    // Valid DAG: v1 → v2 → v3
    dag.add_edge("v1", "v2");
    dag.add_edge("v2", "v3");
    println!("Has cycle? {}", dag.has_cycle());  // Should be false

    // Add cycle: v3 → v1
    dag.add_edge("v3", "v1");
    println!("Has cycle? {}", dag.has_cycle());  // Should be true
}
