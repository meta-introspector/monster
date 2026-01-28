use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Serialize)]
struct CoOccurrenceGraph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    stats: GraphStats,
}

#[derive(Debug, Serialize)]
struct Node {
    id: usize,
    ngram: Vec<u8>,
    frequency: usize,
    prime_signature: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct Edge {
    from: usize,
    to: usize,
    weight: usize,
    distance: usize,
}

#[derive(Debug, Serialize)]
struct GraphStats {
    total_nodes: usize,
    total_edges: usize,
    max_degree: usize,
    monster_clusters: usize,
}

const MONSTER_PRIMES: [u32; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎪 N-gram Co-occurrence Graph Builder");
    println!("=====================================\n");
    
    // Load image patterns
    let patterns_data = fs::read_to_string("IMAGE_PATTERNS.json")?;
    let patterns: serde_json::Value = serde_json::from_str(&patterns_data)?;
    
    println!("Building graph from image patterns...\n");
    
    let mut graph = CoOccurrenceGraph {
        nodes: Vec::new(),
        edges: Vec::new(),
        stats: GraphStats {
            total_nodes: 0,
            total_edges: 0,
            max_degree: 0,
            monster_clusters: 0,
        },
    };
    
    let mut ngram_to_id: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut co_occurrences: HashMap<(usize, usize), usize> = HashMap::new();
    
    // Extract n-grams from patterns
    if let Some(images) = patterns.as_array() {
        for image in images {
            if let Some(byte_patterns) = image["byte_patterns"].as_array() {
                let mut prev_id = None;
                
                for pattern in byte_patterns {
                    let bytes = pattern["bytes"].as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u8)
                        .collect::<Vec<_>>();
                    
                    // Get or create node
                    let id = *ngram_to_id.entry(bytes.clone()).or_insert_with(|| {
                        let new_id = graph.nodes.len();
                        graph.nodes.push(Node {
                            id: new_id,
                            ngram: bytes.clone(),
                            frequency: 0,
                            prime_signature: extract_primes(&bytes),
                        });
                        new_id
                    });
                    
                    graph.nodes[id].frequency += 1;
                    
                    // Create edge from previous n-gram (co-occurrence)
                    if let Some(prev) = prev_id {
                        *co_occurrences.entry((prev, id)).or_insert(0) += 1;
                    }
                    
                    prev_id = Some(id);
                }
            }
        }
    }
    
    // Build edges from co-occurrences
    for ((from, to), weight) in co_occurrences {
        graph.edges.push(Edge {
            from,
            to,
            weight,
            distance: 1,
        });
    }
    
    // Calculate stats
    graph.stats.total_nodes = graph.nodes.len();
    graph.stats.total_edges = graph.edges.len();
    
    // Find max degree
    let mut degrees: HashMap<usize, usize> = HashMap::new();
    for edge in &graph.edges {
        *degrees.entry(edge.from).or_insert(0) += 1;
        *degrees.entry(edge.to).or_insert(0) += 1;
    }
    graph.stats.max_degree = *degrees.values().max().unwrap_or(&0);
    
    // Find Monster clusters (nodes with Monster prime signatures)
    graph.stats.monster_clusters = graph.nodes.iter()
        .filter(|n| !n.prime_signature.is_empty())
        .count();
    
    println!("📊 Graph Statistics:");
    println!("  Nodes: {}", graph.stats.total_nodes);
    println!("  Edges: {}", graph.stats.total_edges);
    println!("  Max degree: {}", graph.stats.max_degree);
    println!("  Monster clusters: {}", graph.stats.monster_clusters);
    
    // Find most connected nodes
    println!("\n🔗 Top 10 Most Connected Nodes:");
    let mut node_degrees: Vec<_> = degrees.iter().collect();
    node_degrees.sort_by_key(|(_, d)| std::cmp::Reverse(**d));
    
    for (node_id, degree) in node_degrees.iter().take(10) {
        let node = &graph.nodes[**node_id];
        println!("  Node {}: degree={}, primes={:?}, freq={}",
                 node_id, degree, node.prime_signature, node.frequency);
    }
    
    // Export graph
    fs::write("GRAPH.json", serde_json::to_string_pretty(&graph)?)?;
    
    // Export DOT format for visualization
    let dot = generate_dot(&graph);
    fs::write("GRAPH.dot", dot)?;
    
    println!("\n✓ Graph built!");
    println!("\n📁 Output:");
    println!("  GRAPH.json - Full graph data");
    println!("  GRAPH.dot - Graphviz format");
    println!("\nVisualize with: dot -Tpng GRAPH.dot -o graph.png");
    
    Ok(())
}

fn extract_primes(bytes: &[u8]) -> Vec<u32> {
    let mut primes = Vec::new();
    for &byte in bytes {
        for &prime in &MONSTER_PRIMES {
            if byte as u32 % prime == 0 && !primes.contains(&prime) {
                primes.push(prime);
            }
        }
    }
    primes
}

fn generate_dot(graph: &CoOccurrenceGraph) -> String {
    let mut dot = String::from("digraph MonsterGraph {\n");
    dot.push_str("  rankdir=LR;\n");
    dot.push_str("  node [shape=circle];\n\n");
    
    // Add nodes (limit to top 100 by frequency)
    let mut nodes: Vec<_> = graph.nodes.iter().collect();
    nodes.sort_by_key(|n| std::cmp::Reverse(n.frequency));
    
    for node in nodes.iter().take(100) {
        let color = if !node.prime_signature.is_empty() {
            "red"
        } else {
            "lightblue"
        };
        dot.push_str(&format!("  {} [label=\"{}\", fillcolor={}, style=filled];\n",
                             node.id, node.frequency, color));
    }
    
    dot.push_str("\n");
    
    // Add edges (limit to weight > 1)
    for edge in &graph.edges {
        if edge.weight > 1 && edge.from < 100 && edge.to < 100 {
            dot.push_str(&format!("  {} -> {} [label=\"{}\"];\n",
                                 edge.from, edge.to, edge.weight));
        }
    }
    
    dot.push_str("}\n");
    dot
}
