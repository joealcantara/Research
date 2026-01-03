use std::collections::HashMap;
use std::collections::HashSet;


pub struct DAG {
    adjacency_list: HashMap<String, Vec<String>>,
}

impl DAG {
    pub fn new() -> Self {
        DAG {
            adjacency_list: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, from: &str, to: &str) {
        self.adjacency_list
            .entry(from.to_string())
            .or_insert_with(Vec::new)
            .push(to.to_string());
    }

    pub fn has_cycle(&self) -> bool {
        // TODO: implement
        let mut visiting: HashSet<String> = HashSet::new();
        let mut visited: HashSet<String> = HashSet::new();

        for node in self.adjacency_list.keys(){
            if !visited.contains(node){
               if self.dfs(node, &mut visiting, &mut visited) {
                   return true;
                }
            }
        }
    false
    }

    fn dfs(&self, node: &str, visiting: &mut HashSet<String>, visited: &mut HashSet<String>) -> bool {
      visiting.insert(node.to_string());

      if let Some(children) = self.adjacency_list.get(node) {
          for child in children {
              // If child is visiting → cycle!
              if visiting.contains(child) {
                  return true;
              }

              // If child not visited, recursively check
              if !visited.contains(child) {
                  if self.dfs(child, visiting, visited) {
                      return true;  // Cycle found in recursion
                  }
              }
          }
      }

      visiting.remove(node);
      visited.insert(node.to_string());

      // 4. No cycle found
      false

    }
}

/// Check if a structure (HashMap representation) is a valid DAG (no cycles).
///
/// # Arguments
/// * `structure` - HashMap mapping variables to their parent lists
///
/// # Returns
/// true if the structure is a DAG (no cycles), false otherwise
pub fn is_dag(structure: &HashMap<String, Vec<String>>) -> bool {
    let mut dag = DAG::new();

    // Build DAG from structure
    // Structure maps child -> parents, DAG expects parent -> children
    for (child, parents) in structure {
        for parent in parents {
            dag.add_edge(parent, child);
        }
    }

    !dag.has_cycle()
}


