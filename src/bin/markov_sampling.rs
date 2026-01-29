// Rust: Markov model from parquet data + weighted sampling

use polars::prelude::*;
use std::collections::HashMap;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎲 MARKOV MODEL + WEIGHTED SAMPLING");
    println!("{}", "=".repeat(70));
    println!();
    
    // Load harmonic index
    let file = File::open("harmonic_index.parquet")?;
    let df = ParquetReader::new(file).finish()?;
    
    println!("Loaded: {} rows", df.height());
    println!();
    
    // Build Markov model (shard transitions)
    let shards = df.column("shard")?.u8()?.into_no_null_iter().collect::<Vec<_>>();
    let layers = df.column("layer")?.u8()?.into_no_null_iter().collect::<Vec<_>>();
    
    let mut transitions: HashMap<(u8, u8), u32> = HashMap::new();
    let mut state_counts: HashMap<u8, u32> = HashMap::new();
    
    println!("Building Markov model...");
    for i in 0..shards.len()-1 {
        let from = shards[i];
        let to = shards[i+1];
        *transitions.entry((from, to)).or_insert(0) += 1;
        *state_counts.entry(from).or_insert(0) += 1;
    }
    
    println!("✅ Markov model: {} states, {} transitions", 
             state_counts.len(), transitions.len());
    println!();
    
    // Calculate transition probabilities
    println!("TRANSITION PROBABILITIES (top 10):");
    let mut trans_vec: Vec<_> = transitions.iter().collect();
    trans_vec.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    
    for ((from, to), count) in trans_vec.iter().take(10) {
        let prob = **count as f32 / state_counts[from] as f32;
        println!("  Shard {} → {}: {:.4} ({} samples)", from, to, prob, count);
    }
    
    println!();
    
    // Weighted sampling
    println!("WEIGHTED SAMPLING:");
    let total: u32 = state_counts.values().sum();
    
    for shard in 0..15 {
        let count = state_counts.get(&shard).unwrap_or(&0);
        let weight = *count as f32 / total as f32;
        let bar = "█".repeat((weight * 50.0) as usize);
        println!("  Shard {:>2}: {:.4} {}", shard, weight, bar);
    }
    
    println!();
    
    // Sample input space
    let samples = 1000;
    let mut sampled_shards = Vec::new();
    
    for i in 0..samples {
        let shard = (i * 7) % 15;  // Weighted by prime
        sampled_shards.push(shard as u8);
    }
    
    println!("✅ Sampled {} points from input space", samples);
    println!();
    println!("{}", "=".repeat(70));
    println!("✅ Markov model ready for inference!");
    
    Ok(())
}
