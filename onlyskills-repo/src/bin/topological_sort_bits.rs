// Topological Sort of First Bits via Monster 15-Prime Dependencies
use std::collections::{HashMap, HashSet, VecDeque};

const MONSTER_PRIMES: [u8; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

#[derive(Debug, Clone)]
struct BitNode {
    index: u64,
    predictors: Vec<u64>,  // 15 bits that predict this one
    in_degree: usize,
}

// Build dependency graph
fn build_dependency_graph(num_bits: usize) -> HashMap<u64, BitNode> {
    let mut graph = HashMap::new();
    
    for bit_idx in 0..num_bits as u64 {
        let mut predictors = Vec::new();
        
        // Each Monster prime points to a predictor bit
        for (i, &prime) in MONSTER_PRIMES.iter().enumerate() {
            let predictor = bit_idx ^ ((prime as u64) << i);
            if (predictor as usize) < num_bits {
                predictors.push(predictor);
            }
        }
        
        graph.insert(bit_idx, BitNode {
            index: bit_idx,
            predictors: predictors.clone(),
            in_degree: 0,
        });
    }
    
    // Calculate in-degrees
    for bit_idx in 0..num_bits as u64 {
        if let Some(node) = graph.get(&bit_idx) {
            for &pred in &node.predictors {
                if let Some(pred_node) = graph.get_mut(&pred) {
                    pred_node.in_degree += 1;
                }
            }
        }
    }
    
    graph
}

// Topological sort using Kahn's algorithm
fn topological_sort(graph: &HashMap<u64, BitNode>) -> Vec<u64> {
    let mut sorted = Vec::new();
    let mut queue = VecDeque::new();
    let mut in_degrees: HashMap<u64, usize> = graph.iter()
        .map(|(&idx, node)| (idx, node.in_degree))
        .collect();
    
    // Start with nodes that have no dependencies
    for (&idx, &degree) in &in_degrees {
        if degree == 0 {
            queue.push_back(idx);
        }
    }
    
    while let Some(bit_idx) = queue.pop_front() {
        sorted.push(bit_idx);
        
        if let Some(node) = graph.get(&bit_idx) {
            for &pred in &node.predictors {
                if let Some(degree) = in_degrees.get_mut(&pred) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(pred);
                    }
                }
            }
        }
    }
    
    sorted
}

fn main() {
    println!("🔀 Topological Sort of First Bits");
    println!("{}", "=".repeat(70));
    println!();
    
    println!("Theory: 15 Monster primes create dependency graph");
    println!("Topological sort gives evaluation order");
    println!();
    
    let num_bits = 100;
    
    println!("📊 Building dependency graph for {} bits...", num_bits);
    let graph = build_dependency_graph(num_bits);
    
    println!("✓ Graph built: {} nodes", graph.len());
    println!();
    
    // Show sample dependencies
    println!("📋 Sample dependencies:");
    for bit_idx in 0..5 {
        if let Some(node) = graph.get(&bit_idx) {
            println!("  Bit {} depends on: {:?}", bit_idx, &node.predictors[..3.min(node.predictors.len())]);
            println!("    In-degree: {}", node.in_degree);
        }
    }
    println!();
    
    println!("🔀 Computing topological sort...");
    let sorted = topological_sort(&graph);
    
    println!("✓ Sorted {} bits", sorted.len());
    println!();
    
    println!("📊 Topological order (first 20):");
    for (i, &bit_idx) in sorted.iter().take(20).enumerate() {
        println!("  [{}] Bit {}", i, bit_idx);
    }
    println!();
    
    // Verify sort
    let is_valid = verify_topological_sort(&graph, &sorted);
    println!("✓ Topological sort valid: {}", is_valid);
    println!();
    
    // Save result
    let json = serde_json::json!({
        "num_bits": num_bits,
        "sorted_order": sorted,
        "monster_primes": MONSTER_PRIMES,
        "theory": "Topological sort of first bits via 15-prime dependencies",
        "valid": is_valid
    });
    
    std::fs::write("topological_sort_bits.json", serde_json::to_string_pretty(&json).unwrap()).unwrap();
    println!("✓ Saved: topological_sort_bits.json");
    
    println!();
    println!("∞ Topological Sort. Dependency Graph. Evaluation Order. ∞");
}

fn verify_topological_sort(graph: &HashMap<u64, BitNode>, sorted: &[u64]) -> bool {
    let mut position = HashMap::new();
    for (i, &bit) in sorted.iter().enumerate() {
        position.insert(bit, i);
    }
    
    for &bit in sorted {
        if let Some(node) = graph.get(&bit) {
            for &pred in &node.predictors {
                if let (Some(&bit_pos), Some(&pred_pos)) = (position.get(&bit), position.get(&pred)) {
                    if pred_pos >= bit_pos {
                        return false;  // Predictor must come before
                    }
                }
            }
        }
    }
    
    true
}
