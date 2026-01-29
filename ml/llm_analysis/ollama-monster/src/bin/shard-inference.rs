//! Simple inference test: Feed prompt through shards

use anyhow::Result;
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("🔮 SHARD INFERENCE TEST");
    println!("=======================\n");
    
    // Load all shards
    println!("Loading shards...");
    let mut shards = HashMap::new();
    
    for n in 1..=71 {
        let filename = format!("shards/qwen2.5-3b-shard-{}.gguf", n);
        if let Ok(neurons) = load_neurons(&filename) {
            shards.insert(n, neurons);
            if n <= 5 || n == 47 || n == 71 {
                println!("  Shard {}: {} neurons", n, neurons.len());
            }
        }
    }
    
    println!("\n✓ Loaded {} shards\n", shards.len());
    
    // Test prompts
    let prompts = [
        "Monster group",
        "John Conway", 
        "Binary logic",
        "Prime numbers",
    ];
    
    for prompt in prompts {
        println!("📝 Prompt: \"{}\"", prompt);
        
        // Calculate resonance with each shard
        let mut resonances: Vec<(u32, f64)> = shards.iter()
            .map(|(&n, neurons)| {
                let score = calculate_resonance(prompt, neurons, n);
                (n, score)
            })
            .collect();
        
        resonances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("  Top resonating shards:");
        for (n, score) in resonances.iter().take(5) {
            println!("    Shard {}: {:.3}", n, score);
        }
        println!();
    }
    
    Ok(())
}

fn load_neurons(filename: &str) -> Result<Vec<f32>> {
    let mut file = File::open(filename)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    let data_start = 100;
    let mut neurons = Vec::new();
    let mut offset = data_start;
    
    while offset + 4 <= buffer.len() {
        let bytes = [
            buffer[offset],
            buffer[offset + 1],
            buffer[offset + 2],
            buffer[offset + 3],
        ];
        let value = f32::from_le_bytes(bytes);
        
        if value.is_finite() && value.abs() < 1.0 {
            neurons.push(value);
        }
        
        offset += 4;
    }
    
    Ok(neurons)
}

fn calculate_resonance(prompt: &str, neurons: &[f32], shard_n: u32) -> f64 {
    if neurons.is_empty() {
        return 0.0;
    }
    
    // Simple heuristic: prompt hash mod shard number
    let prompt_hash: u32 = prompt.bytes().map(|b| b as u32).sum();
    let alignment = (prompt_hash % shard_n) as f64 / shard_n as f64;
    
    // Combine with neuron statistics
    let avg = neurons.iter().sum::<f32>() / neurons.len() as f32;
    let neuron_score = avg.abs() as f64;
    
    // Weight by shard size (larger shards = more important)
    let size_weight = (neurons.len() as f64).ln();
    
    alignment * neuron_score * size_weight
}
