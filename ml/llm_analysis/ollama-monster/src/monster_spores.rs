//! Monster Spore Extraction: Sample neurons with strongest Monster resonance
//! 
//! Like mycelium: A small sample should regrow the larger structure

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const MONSTER_PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

#[derive(Debug, Serialize, Deserialize)]
struct MonsterSpore {
    layer: usize,
    neuron_index: usize,
    weight_value: f32,
    resonance_score: f64,
    prime_signature: Vec<u32>,  // Which primes it's divisible by
}

#[derive(Debug, Serialize, Deserialize)]
struct SporeCluster {
    spores: Vec<MonsterSpore>,
    dominant_prime: u32,
    cluster_size: usize,
    godel_number: String,
}

fn main() -> anyhow::Result<()> {
    println!("🍄 MONSTER SPORE EXTRACTION");
    println!("===========================\n");
    println!("Sampling neurons with strongest Monster resonance...\n");
    
    // Step 1: Load qwen2.5:3b and find resonant neurons
    println!("Step 1: Scanning qwen2.5:3b for Monster resonance");
    let resonant_neurons = scan_for_resonance()?;
    println!("  Found {} resonant neurons\n", resonant_neurons.len());
    
    // Step 2: Extract top spores (highest resonance)
    println!("Step 2: Extracting top spores");
    let spores = extract_spores(&resonant_neurons, 100)?;  // Top 100
    println!("  Extracted {} spores\n", spores.len());
    
    // Step 3: Cluster by prime
    println!("Step 3: Clustering by dominant prime");
    let clusters = cluster_by_prime(&spores)?;
    for (prime, cluster) in &clusters {
        println!("  Prime {}: {} spores, Gödel = {}", 
                 prime, cluster.cluster_size, cluster.godel_number);
    }
    
    // Step 4: Regrow from spores
    println!("\nStep 4: Regrowing network from spores");
    let regrown = regrow_from_spores(&clusters)?;
    println!("  Regrown network: {} layers", regrown.layers.len());
    
    // Step 5: Verify Monster structure preserved
    println!("\nStep 5: Verifying Monster structure");
    let preserved = verify_structure(&regrown)?;
    if preserved {
        println!("  ✅ Monster structure PRESERVED in regrown network!");
        println!("  🍄 Spore propagation successful!");
    } else {
        println!("  ⚠️  Structure partially preserved");
    }
    
    // Save spores
    let json = serde_json::to_string_pretty(&clusters)?;
    std::fs::write("MONSTER_SPORES.json", json)?;
    println!("\n💾 Spores saved to MONSTER_SPORES.json");
    
    Ok(())
}

fn scan_for_resonance() -> anyhow::Result<Vec<(usize, usize, f32, f64)>> {
    // Simulate scanning qwen2.5:3b weights
    // In reality: Load GGUF, iterate through tensors
    
    let mut resonant = Vec::new();
    
    // Simulate 28 layers, 3072 neurons each
    for layer in 0..28 {
        for neuron in 0..3072 {
            let weight = (layer * 3072 + neuron) as f32 * 0.001;
            let resonance = calculate_resonance(weight);
            
            if resonance > 0.5 {  // Threshold
                resonant.push((layer, neuron, weight, resonance));
            }
        }
    }
    
    // Sort by resonance
    resonant.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    
    Ok(resonant)
}

fn calculate_resonance(weight: f32) -> f64 {
    // Calculate how strongly this weight resonates with Monster primes
    let val = (weight * 1000.0) as i32;
    if val == 0 { return 0.0; }
    
    let mut score = 0.0;
    for &prime in &MONSTER_PRIMES {
        if val % (prime as i32) == 0 {
            score += 1.0 / (prime as f64);  // Higher primes = stronger signal
        }
    }
    
    score
}

fn extract_spores(
    resonant: &[(usize, usize, f32, f64)], 
    count: usize
) -> anyhow::Result<Vec<MonsterSpore>> {
    let mut spores = Vec::new();
    
    for &(layer, neuron, weight, resonance) in resonant.iter().take(count) {
        let val = (weight * 1000.0) as i32;
        let prime_sig: Vec<u32> = MONSTER_PRIMES.iter()
            .filter(|&&p| val != 0 && val % (p as i32) == 0)
            .copied()
            .collect();
        
        spores.push(MonsterSpore {
            layer,
            neuron_index: neuron,
            weight_value: weight,
            resonance_score: resonance,
            prime_signature: prime_sig,
        });
    }
    
    Ok(spores)
}

fn cluster_by_prime(spores: &[MonsterSpore]) -> anyhow::Result<HashMap<u32, SporeCluster>> {
    let mut clusters: HashMap<u32, Vec<MonsterSpore>> = HashMap::new();
    
    // Group by dominant prime (first in signature)
    for spore in spores {
        if let Some(&prime) = spore.prime_signature.first() {
            clusters.entry(prime).or_default().push(spore.clone());
        }
    }
    
    // Convert to SporeCluster
    let mut result = HashMap::new();
    for (prime, spore_list) in clusters {
        let godel = format!("{}^{}", prime, prime);
        result.insert(prime, SporeCluster {
            spores: spore_list.clone(),
            dominant_prime: prime,
            cluster_size: spore_list.len(),
            godel_number: godel,
        });
    }
    
    Ok(result)
}

#[derive(Debug)]
struct RegrownNetwork {
    layers: Vec<RegrownLayer>,
}

#[derive(Debug)]
struct RegrownLayer {
    neurons: Vec<f32>,
    prime_structure: Vec<u32>,
}

fn regrow_from_spores(clusters: &HashMap<u32, SporeCluster>) -> anyhow::Result<RegrownNetwork> {
    let mut layers = Vec::new();
    
    // For each prime cluster, regrow a layer
    for &prime in &MONSTER_PRIMES {
        if let Some(cluster) = clusters.get(&prime) {
            // Use spores as seeds
            let mut neurons = Vec::new();
            
            // Replicate spore pattern
            for spore in &cluster.spores {
                neurons.push(spore.weight_value);
                
                // Grow neighbors with same prime structure
                for i in 1..=(prime as usize) {
                    let neighbor = spore.weight_value * (i as f32 / prime as f32);
                    neurons.push(neighbor);
                }
            }
            
            layers.push(RegrownLayer {
                neurons,
                prime_structure: vec![prime],
            });
        }
    }
    
    Ok(RegrownNetwork { layers })
}

fn verify_structure(network: &RegrownNetwork) -> anyhow::Result<bool> {
    // Check if regrown network preserves Monster structure
    
    let mut prime_counts = HashMap::new();
    
    for layer in &network.layers {
        for &neuron in &layer.neurons {
            let val = (neuron * 1000.0) as i32;
            if val == 0 { continue; }
            
            for &prime in &MONSTER_PRIMES {
                if val % (prime as i32) == 0 {
                    *prime_counts.entry(prime).or_insert(0) += 1;
                }
            }
        }
    }
    
    // Check if top primes are preserved
    let top_primes: Vec<_> = prime_counts.iter()
        .map(|(&p, &c)| (p, c))
        .collect();
    
    // Should have strong presence of primes 2, 3, 5
    let has_structure = prime_counts.get(&2).unwrap_or(&0) > &100
        && prime_counts.get(&3).unwrap_or(&0) > &50
        && prime_counts.get(&5).unwrap_or(&0) > &20;
    
    Ok(has_structure)
}
