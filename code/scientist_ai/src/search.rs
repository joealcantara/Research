use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use rand::prelude::*;
use crate::inference::{compute_posteriors, ScoringMethod, StructureScore};
use crate::dag::is_dag;

/// Generate neighbor structures by adding, removing, or reversing single edges.
///
/// For each variable, tries adding edges from other variables (if doesn't create cycle),
/// removing existing edges, or reversing existing edges (if doesn't create cycle).
///
/// # Arguments
/// * `structure` - Current DAG structure
/// * `variables` - List of all variable names in the dataset
///
/// # Returns
/// Vector of neighbor DAG structures
pub fn generate_neighbors(
    structure: &HashMap<String, Vec<String>>,
    variables: &[String],
) -> Vec<HashMap<String, Vec<String>>> {
    let mut neighbors = Vec::new();

    for var in variables {
        let current_parents = structure.get(var).cloned().unwrap_or_default();

        // Try adding each possible parent
        for potential_parent in variables {
            if potential_parent == var {
                continue; // Can't be own parent
            }
            if current_parents.contains(potential_parent) {
                continue; // Already a parent
            }

            // Try adding this edge
            let mut new_structure = structure.clone();
            let mut new_parents = current_parents.clone();
            new_parents.push(potential_parent.clone());
            new_structure.insert(var.clone(), new_parents);

            // Check if still a DAG (no cycles)
            if is_dag(&new_structure) {
                neighbors.push(new_structure);
            }
        }

        // Try removing each existing parent
        for parent_to_remove in &current_parents {
            let mut new_structure = structure.clone();
            let new_parents: Vec<String> = current_parents
                .iter()
                .filter(|p| *p != parent_to_remove)
                .cloned()
                .collect();
            new_structure.insert(var.clone(), new_parents);
            neighbors.push(new_structure);
        }

        // Try reversing each edge (child <- parent becomes child -> parent)
        for parent in &current_parents {
            // Remove parent -> var edge
            let mut new_structure = structure.clone();
            let new_parents: Vec<String> = current_parents
                .iter()
                .filter(|p| *p != parent)
                .cloned()
                .collect();
            new_structure.insert(var.clone(), new_parents);

            // Add var -> parent edge (reverse)
            let parent_parents = new_structure.get(parent).cloned().unwrap_or_default();
            let mut new_parent_parents = parent_parents.clone();
            if !new_parent_parents.contains(var) {
                new_parent_parents.push(var.clone());
                new_structure.insert(parent.clone(), new_parent_parents);

                // Check if still a DAG
                if is_dag(&new_structure) {
                    neighbors.push(new_structure);
                }
            }
        }
    }

    // Deduplicate neighbors
    let mut unique_neighbors = Vec::new();
    let mut seen = HashSet::new();

    for neighbor in neighbors {
        // Create a canonical representation for comparison
        let mut edges: Vec<(String, String)> = Vec::new();
        for (child, parents) in &neighbor {
            for parent in parents {
                edges.push((parent.clone(), child.clone()));
            }
        }
        edges.sort();
        let key = format!("{:?}", edges);

        if !seen.contains(&key) {
            seen.insert(key);
            unique_neighbors.push(neighbor);
        }
    }

    unique_neighbors
}

/// Sample structures proportionally to their posterior probabilities.
///
/// Uses softmax sampling with optional temperature to control exploration.
///
/// # Arguments
/// * `structures` - HashMap of named structures with their scores
/// * `k` - Number of structures to sample
/// * `temperature` - Controls exploration (higher = more uniform, default: 1.0)
/// * `rng` - Random number generator
///
/// # Returns
/// Vector of sampled structure names
pub fn sample_structures(
    structures: &HashMap<String, StructureScore>,
    k: usize,
    temperature: f64,
    rng: &mut impl Rng,
) -> Vec<String> {
    if structures.len() <= k {
        return structures.keys().cloned().collect();
    }

    // Extract names and posteriors
    let names: Vec<String> = structures.keys().cloned().collect();
    let posteriors: Vec<f64> = names
        .iter()
        .map(|name| structures[name].posterior)
        .collect();

    // Apply temperature (optional, posteriors are already normalized)
    let probs: Vec<f64> = if (temperature - 1.0).abs() < 1e-6 {
        posteriors
    } else {
        let log_probs: Vec<f64> = posteriors.iter().map(|p| p.ln() / temperature).collect();
        let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_probs: Vec<f64> = log_probs.iter().map(|lp| (lp - max_log_prob).exp()).collect();
        let sum: f64 = exp_probs.iter().sum();
        exp_probs.iter().map(|p| p / sum).collect()
    };

    // Sample without replacement using weighted sampling
    let mut selected = HashSet::new();
    let mut sampled = Vec::new();

    while sampled.len() < k && sampled.len() < names.len() {
        // Build list of available (not yet selected) items
        let available_indices: Vec<usize> = (0..names.len())
            .filter(|i| !selected.contains(i))
            .collect();

        if available_indices.is_empty() {
            break;
        }

        // Get probabilities for available items and renormalize
        let available_probs: Vec<f64> = available_indices
            .iter()
            .map(|&i| probs[i])
            .collect();

        let prob_sum: f64 = available_probs.iter().sum();

        // Handle edge case: if all probs are zero (shouldn't happen but be safe)
        if prob_sum < 1e-10 {
            // Just pick uniformly from available items
            let idx = available_indices[rng.gen_range(0..available_indices.len())];
            selected.insert(idx);
            sampled.push(names[idx].clone());
            continue;
        }

        let normalized_probs: Vec<f64> = available_probs
            .iter()
            .map(|p| p / prob_sum)
            .collect();

        // Sample from available items with renormalized probabilities
        let r: f64 = rng.gen_range(0.0..1.0);
        let mut cumsum = 0.0;

        for (j, &idx) in available_indices.iter().enumerate() {
            cumsum += normalized_probs[j];
            if r < cumsum || j == available_indices.len() - 1 {
                // Select this item (or last item if we somehow missed due to rounding)
                selected.insert(idx);
                sampled.push(names[idx].clone());
                break;
            }
        }
    }

    sampled
}

/// Enumerate all possible DAG structures for a given set of variables.
///
/// For small datasets (≤5 variables), exhaustive enumeration is feasible.
/// Generates all possible edge combinations and filters to valid DAGs.
///
/// # Arguments
/// * `variables` - List of variable names
///
/// # Returns
/// Vector of all possible DAG structures
///
/// # Note
/// The number of DAGs grows super-exponentially:
/// - 3 variables: 25 DAGs
/// - 4 variables: 543 DAGs
/// - 5 variables: 29,281 DAGs
/// - 6+ variables: not recommended (millions of DAGs)
pub fn enumerate_all_dags(variables: &[String]) -> Vec<HashMap<String, Vec<String>>> {
    let n = variables.len();

    // Generate all possible directed edges
    let mut all_edges = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                // Edge from variables[i] to variables[j]
                all_edges.push((i, j));
            }
        }
    }

    let num_possible_edges = all_edges.len();
    let num_subsets = 1 << num_possible_edges; // 2^num_edges

    let mut all_dags = Vec::new();

    // Enumerate all subsets of edges (all possible graphs)
    for subset in 0..num_subsets {
        let mut structure = HashMap::new();

        // Initialize all variables with empty parent lists
        for var in variables {
            structure.insert(var.clone(), vec![]);
        }

        // Add edges corresponding to this subset
        for (edge_idx, &(parent_idx, child_idx)) in all_edges.iter().enumerate() {
            if (subset >> edge_idx) & 1 == 1 {
                // This edge is included in the subset
                let parent = variables[parent_idx].clone();
                let child = variables[child_idx].clone();

                structure.entry(child.clone())
                    .or_insert_with(Vec::new)
                    .push(parent);
            }
        }

        // Check if this structure is a DAG (acyclic)
        if is_dag(&structure) {
            all_dags.push(structure);
        }
    }

    all_dags
}

/// Stochastic beam search for DAG structure learning.
///
/// Iteratively explores the space of DAG structures by:
/// 1. Generating neighbors of current beam structures
/// 2. Scoring all candidates
/// 3. Sampling new beam proportionally to posteriors
/// 4. Repeating until convergence or max rounds
///
/// # Arguments
/// * `df` - DataFrame with variables
/// * `variables` - List of variable names to include in structure
/// * `beam_size` - Number of structures to keep in beam (default: 10)
/// * `max_rounds` - Maximum search iterations (default: 10)
/// * `method` - Scoring method (EdgeBased, BIC, BDeu)
/// * `temperature` - Sampling temperature for exploration (default: 1.0)
/// * `verbose` - Print progress (default: true)
///
/// # Returns
/// Best structure found (HashMap mapping variables to parent lists)
pub fn stochastic_beam_search(
    df: &DataFrame,
    variables: &[String],
    beam_size: usize,
    max_rounds: usize,
    method: &ScoringMethod,
    temperature: f64,
    verbose: bool,
) -> Result<HashMap<String, Vec<String>>, PolarsError> {
    let mut rng = rand::thread_rng();

    // Round 0: Initialize with empty structure
    let mut current_beam: Vec<HashMap<String, Vec<String>>> = vec![HashMap::new()];

    // Ensure all variables are represented in structure
    for var in variables {
        current_beam[0].insert(var.clone(), vec![]);
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut best_structure = current_beam[0].clone();

    if verbose {
        println!("Starting stochastic beam search:");
        println!("  Variables: {}", variables.len());
        println!("  Beam size: {}", beam_size);
        println!("  Max rounds: {}", max_rounds);
        println!("  Scoring: {:?}", method);
        println!();
    }

    for round in 0..max_rounds {
        // Generate all neighbors of current beam
        let mut candidates = HashMap::new();

        for (i, structure) in current_beam.iter().enumerate() {
            // Include current structure
            candidates.insert(format!("current_{}", i), structure.clone());

            // Generate neighbors
            let neighbors = generate_neighbors(structure, variables);
            for (j, neighbor) in neighbors.into_iter().enumerate() {
                candidates.insert(format!("neighbor_{}_{}", i, j), neighbor);
            }
        }

        if verbose {
            println!("Round {}: {} candidate structures", round + 1, candidates.len());
        }

        // Score all candidates
        let scores = compute_posteriors(df, &candidates, method)?;

        // Find best structure in this round
        let round_best = scores
            .values()
            .max_by(|a, b| a.log_posterior.partial_cmp(&b.log_posterior).unwrap())
            .unwrap();

        if round_best.log_posterior > best_score {
            best_score = round_best.log_posterior;
            // Find which structure had this score
            for (name, score) in &scores {
                if (score.log_posterior - best_score).abs() < 1e-9 {
                    best_structure = candidates[name].clone();
                    break;
                }
            }

            if verbose {
                println!("  New best score: {:.3} (edges: {})", best_score, round_best.edges);
            }
        } else if verbose {
            println!("  Best score: {:.3} (no improvement)", best_score);
        }

        // Sample new beam
        let sampled_names = sample_structures(&scores, beam_size, temperature, &mut rng);
        current_beam = sampled_names
            .iter()
            .map(|name| candidates[name].clone())
            .collect();

        // Check for convergence (simplified: just check if no improvement)
        // In practice, could check if beam is stable
    }

    if verbose {
        println!("\nSearch complete!");
        println!("Best structure score: {:.3}", best_score);
        let edge_count: usize = best_structure.values().map(|p| p.len()).sum();
        println!("Edges: {}", edge_count);
    }

    Ok(best_structure)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    #[test]
    fn test_enumerate_all_dags() {
        println!("\n=== Testing Exhaustive DAG Enumeration ===\n");

        let variables = vec!["X1".to_string(), "X2".to_string(), "X3".to_string()];

        println!("Enumerating all DAGs for 3 variables...");
        let all_dags = enumerate_all_dags(&variables);

        println!("Found {} unique DAGs", all_dags.len());
        println!("Expected: 25 DAGs for 3 variables");

        // Verify we get exactly 25 DAGs for 3 variables (known result)
        assert_eq!(all_dags.len(), 25, "Should enumerate exactly 25 DAGs for 3 variables");

        // Print first few examples
        println!("\nFirst 5 DAGs:");
        for (i, dag) in all_dags.iter().take(5).enumerate() {
            let edge_count: usize = dag.values().map(|p| p.len()).sum();
            println!("  DAG {}: {} edges", i + 1, edge_count);
            for (var, parents) in dag {
                if !parents.is_empty() {
                    println!("    {} ← {:?}", var, parents);
                }
            }
        }

        // Verify all are valid DAGs
        for dag in &all_dags {
            assert!(is_dag(dag), "All enumerated structures should be DAGs");
        }

        println!("\n✓ All structures are valid DAGs");
        println!("\n=== Exhaustive Enumeration Test Passed! ===");
    }

    #[test]
    fn test_generate_neighbors() {
        println!("\n=== Testing Neighbor Generation ===\n");

        // Start with empty structure for 3 variables
        let mut structure = HashMap::new();
        structure.insert("X1".to_string(), vec![]);
        structure.insert("X2".to_string(), vec![]);
        structure.insert("X3".to_string(), vec![]);

        let variables = vec!["X1".to_string(), "X2".to_string(), "X3".to_string()];

        // Generate neighbors
        let neighbors = generate_neighbors(&structure, &variables);

        println!("Starting structure: empty (no edges)");
        println!("Generated {} neighbors", neighbors.len());

        // For 3 variables starting from empty:
        // Each variable can have 2 other variables as parents
        // Total: 3 variables × 2 parents = 6 possible single-edge additions
        assert!(!neighbors.is_empty());
        println!("  (Each neighbor adds one edge)");

        // Now test from a structure with edges
        let mut structure_with_edge = HashMap::new();
        structure_with_edge.insert("X1".to_string(), vec![]);
        structure_with_edge.insert("X2".to_string(), vec!["X1".to_string()]);
        structure_with_edge.insert("X3".to_string(), vec![]);

        let neighbors_2 = generate_neighbors(&structure_with_edge, &variables);
        println!("\nStarting structure: X1 → X2");
        println!("Generated {} neighbors", neighbors_2.len());
        println!("  (Can add more edges, remove X1→X2, or reverse to X2→X1)");

        println!("\n=== Neighbor Generation Test Passed! ===");
    }

    #[test]
    fn test_stochastic_beam_search() {
        println!("\n=== Testing Stochastic Beam Search ===\n");

        // Create simple test data with clear structure: X1 → X2 → X3
        let test_df = df![
            "X1" => &[1i64, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            "X2" => &[1i64, 1, 1, 0, 1, 0, 0, 0, 1, 0],
            "X3" => &[1i64, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        ].unwrap();

        let variables = vec!["X1".to_string(), "X2".to_string(), "X3".to_string()];

        println!("Test data: 10 samples, 3 binary variables");
        println!("True structure: X1 → X2 → X3 (chain)\n");

        // Run search with small beam and few rounds for testing
        let method = ScoringMethod::EdgeBased(2.0);
        let result = stochastic_beam_search(
            &test_df,
            &variables,
            5,  // beam_size
            3,  // max_rounds
            &method,
            1.0,  // temperature
            true,  // verbose
        ).unwrap();

        println!("\nLearned structure:");
        for (var, parents) in &result {
            if parents.is_empty() {
                println!("  {} (no parents)", var);
            } else {
                println!("  {} ← {:?}", var, parents);
            }
        }

        // Verify it's a valid DAG
        assert!(is_dag(&result));
        println!("\n✓ Result is a valid DAG");

        // Count edges
        let edges: usize = result.values().map(|p| p.len()).sum();
        println!("✓ Total edges: {}", edges);

        println!("\n=== Search Test Passed! ===");
    }
}
