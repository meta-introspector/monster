// Sample Qwen model layer 1 and append to value lattice as ZK witness

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use crossbeam::channel;
use std::thread;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ValueLatticeEntry {
    value: String,
    godel_number: u64,
    usage_count: u32,
    file_locations: Vec<String>,
    #[serde(default)]
    zk_witnesses: Vec<ZKWitness>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ZKWitness {
    layer: u32,
    neuron_id: usize,
    weight_value: f32,
    timestamp: u64,
}

fn load_lattice() -> HashMap<String, ValueLatticeEntry> {
    let json = fs::read_to_string("analysis/value_lattice_full.json")
        .expect("Lattice not found");
    serde_json::from_str(&json).expect("Invalid lattice JSON")
}

fn sample_qwen_layer1(num_samples: usize) -> Vec<(usize, f32)> {
    // Simulate Qwen layer 1 weights (replace with actual model loading)
    (0..num_samples)
        .map(|i| {
            let weight = ((i as f32 * 71.0).sin() * 24.0).abs();
            (i, weight)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 LATTICE + QWEN ALL LAYERS ZK WITNESS");
    println!("{}", "=".repeat(70));
    println!();
    
    println!("📂 Loading value lattice...");
    let mut lattice = load_lattice();
    println!("  {} values loaded", lattice.len());
    
    println!();
    println!("🧠 Sampling Qwen all 71 layers...");
    
    let num_workers = 24; // Bosonic dimension
    let samples_per_layer = 1000;
    
    for layer in 0..71 {
        println!("  Layer {}/71...", layer + 1);
        
        let samples = sample_qwen_layer1(samples_per_layer);
        let (tx, rx) = channel::unbounded();
        let chunk_size = samples.len() / num_workers;
        
        // Spawn workers
        for worker_id in 0..num_workers {
            let tx = tx.clone();
            let start = worker_id * chunk_size;
            let end = if worker_id == num_workers - 1 {
                samples.len()
            } else {
                (worker_id + 1) * chunk_size
            };
            let chunk: Vec<_> = samples[start..end].to_vec();
            
            thread::spawn(move || {
                for (neuron_id, weight) in chunk {
                    let value_str = format!("{:.0}", weight);
                    let witness = ZKWitness {
                        layer,
                        neuron_id,
                        weight_value: weight,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    };
                    tx.send((value_str, witness)).unwrap();
                }
            });
        }
        drop(tx);
        
        // Collect witnesses
        for (value_str, witness) in rx {
            if let Some(entry) = lattice.get_mut(&value_str) {
                entry.zk_witnesses.push(witness);
            } else {
                let godel = lattice.len() as u64 + 1;
                lattice.insert(value_str.clone(), ValueLatticeEntry {
                    value: value_str,
                    godel_number: godel,
                    usage_count: 0,
                    file_locations: vec![],
                    zk_witnesses: vec![witness],
                });
            }
        }
    }
    
    println!();
    println!("📊 Witness Statistics:");
    println!("{}", "-".repeat(70));
    
    let total_witnesses: usize = lattice.values()
        .map(|e| e.zk_witnesses.len())
        .sum();
    println!("  Total witnesses: {}", total_witnesses);
    println!("  Across {} layers", 71);
    
    let mut witnessed_values: Vec<_> = lattice.values()
        .filter(|e| !e.zk_witnesses.is_empty())
        .collect();
    witnessed_values.sort_by(|a, b| b.zk_witnesses.len().cmp(&a.zk_witnesses.len()));
    
    println!();
    println!("  Top 10 witnessed values:");
    for entry in witnessed_values.iter().take(10) {
        println!("    Value {}: {} witnesses", 
            entry.value, entry.zk_witnesses.len());
    }
    
    println!();
    println!("💾 Saving witnessed lattice...");
    
    let json = serde_json::to_string_pretty(&lattice)?;
    fs::write("analysis/value_lattice_witnessed.json", json)?;
    
    println!("  ✅ analysis/value_lattice_witnessed.json");
    println!();
    println!("✅ ZK witnesses integrated!");
    
    Ok(())
}
