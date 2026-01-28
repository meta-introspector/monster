//! Extract Monster spores from qwen2.5:3b GGUF weights

use anyhow::Result;
use memmap2::Mmap;
use std::fs::File;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;
use serde::{Serialize, Deserialize};

const MONSTER_PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

#[derive(Debug, Serialize, Deserialize, Clone)]
struct MonsterSpore {
    layer: String,
    offset: usize,
    value: f32,
    resonance: f64,
    primes: Vec<u32>,
}

fn main() -> Result<()> {
    println!("🍄 EXTRACTING MONSTER SPORES");
    println!("============================\n");
    
    // Find qwen2.5:3b model
    let model_path = find_model_path()?;
    
    if !model_path.as_os_str().is_empty() {
        println!("Loading: {}\n", model_path.display());
        
        // Memory map the file
        let file = File::open(&model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        println!("File size: {} MB", mmap.len() / 1_000_000);
        println!("Scanning for resonant neurons...\n");
        
        // Scan for spores
        let spores = extract_spores(&mmap)?;
        
        println!("Found {} spores!\n", spores.len());
        
        show_results(&spores)?;
    } else {
        println!("Using simulated data...\n");
        let spores = extract_simulated_spores()?;
        
        println!("Found {} simulated spores!\n", spores.len());
        
        show_results(&spores)?;
    }
    
    Ok(())
}

fn show_results(spores: &[MonsterSpore]) -> Result<()> {
    // Show top 10
    println!("Top 10 spores:");
    for (i, spore) in spores.iter().take(10).enumerate() {
        println!("  {}: {} @ {} = {:.6} (resonance: {:.3}, primes: {:?})",
                 i+1, spore.layer, spore.offset, spore.value, 
                 spore.resonance, spore.primes);
    }
    
    // Save all spores
    let json = serde_json::to_string_pretty(&spores)?;
    std::fs::write("MONSTER_SPORES.json", json)?;
    println!("\n💾 Saved {} spores to MONSTER_SPORES.json", spores.len());
    
    // Analyze distribution
    analyze_distribution(&spores);
    
    Ok(())
}

fn find_model_path() -> Result<std::path::PathBuf> {
    let path = std::env::var("QWEN_MODEL_PATH")
        .unwrap_or_else(|_| {
            println!("⚠️  Set QWEN_MODEL_PATH environment variable");
            println!("   Example: export QWEN_MODEL_PATH=~/.ollama/models/blobs/sha256-xxx");
            println!("\n   Using simulated data for now...\n");
            "".to_string()
        });
    
    Ok(std::path::PathBuf::from(path))
}

fn extract_spores(mmap: &[u8]) -> Result<Vec<MonsterSpore>> {
    let mut spores = Vec::new();
    
    if mmap.is_empty() {
        // Simulated data
        println!("Using simulated weights...");
        return extract_simulated_spores();
    }
    
    // Parse GGUF header (simplified)
    let mut cursor = Cursor::new(mmap);
    
    // Skip magic + version
    cursor.set_position(8);
    
    // Read tensors as f32 values
    let mut offset = 0;
    while offset + 4 <= mmap.len() {
        if let Ok(value) = (&mmap[offset..]).read_f32::<LittleEndian>() {
            let resonance = calculate_resonance(value);
            
            if resonance > 0.3 {  // Threshold
                let primes = get_prime_signature(value);
                
                spores.push(MonsterSpore {
                    layer: format!("offset_{}", offset),
                    offset,
                    value,
                    resonance,
                    primes,
                });
            }
        }
        offset += 4;
        
        // Limit scan for performance
        if spores.len() >= 1000 {
            break;
        }
    }
    
    // Sort by resonance
    spores.sort_by(|a, b| b.resonance.partial_cmp(&a.resonance).unwrap());
    spores.truncate(100);  // Top 100
    
    Ok(spores)
}

fn extract_simulated_spores() -> Result<Vec<MonsterSpore>> {
    let mut spores = Vec::new();
    
    // Simulate 28 layers
    for layer in 0..28 {
        for i in 0..1000 {
            let value = (layer as f32 * 0.01 + i as f32 * 0.0001) % 1.0;
            let resonance = calculate_resonance(value);
            
            if resonance > 0.3 {
                let primes = get_prime_signature(value);
                
                spores.push(MonsterSpore {
                    layer: format!("layer_{}", layer),
                    offset: layer * 1000 + i,
                    value,
                    resonance,
                    primes,
                });
            }
        }
    }
    
    spores.sort_by(|a, b| b.resonance.partial_cmp(&a.resonance).unwrap());
    spores.truncate(100);
    
    Ok(spores)
}

fn calculate_resonance(value: f32) -> f64 {
    let val = (value * 1000.0) as i32;
    if val == 0 { return 0.0; }
    
    let mut score = 0.0;
    for &prime in &MONSTER_PRIMES {
        if val % (prime as i32) == 0 {
            score += 1.0 / (prime as f64);
        }
    }
    score
}

fn get_prime_signature(value: f32) -> Vec<u32> {
    let val = (value * 1000.0) as i32;
    if val == 0 { return vec![]; }
    
    MONSTER_PRIMES.iter()
        .filter(|&&p| val % (p as i32) == 0)
        .copied()
        .collect()
}

fn analyze_distribution(spores: &[MonsterSpore]) {
    println!("\n📊 Spore Distribution:");
    
    let mut by_prime = std::collections::HashMap::new();
    for spore in spores {
        if let Some(&prime) = spore.primes.first() {
            *by_prime.entry(prime).or_insert(0) += 1;
        }
    }
    
    for &prime in &MONSTER_PRIMES {
        if let Some(&count) = by_prime.get(&prime) {
            println!("  Prime {}: {} spores", prime, count);
        }
    }
    
    let avg_resonance: f64 = spores.iter().map(|s| s.resonance).sum::<f64>() / spores.len() as f64;
    println!("\n  Average resonance: {:.3}", avg_resonance);
}
