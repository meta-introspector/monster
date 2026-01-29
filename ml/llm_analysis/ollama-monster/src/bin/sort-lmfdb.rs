//! Sort LMFDB mathematical objects into 71 Monster shards by prime resonance

use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const MONSTER_PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

fn main() -> Result<()> {
    println!("🔷 Sorting LMFDB by Prime Resonance");
    println!("====================================\n");
    
    let lmfdb_dir = Path::new("/mnt/data1/meta-introspector/data/lmfdb-collected");
    let output_dir = Path::new("monster-shards");
    
    let mut total_stats: HashMap<u32, usize> = HashMap::new();
    
    // Process each JSON file
    for entry in fs::read_dir(lmfdb_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let shards = process_lmfdb_file(&path, output_dir)?;
            
            for (shard, count) in shards {
                *total_stats.entry(shard).or_insert(0) += count;
            }
        }
    }
    
    // Print summary
    println!("\n====================================");
    println!("Summary:\n");
    
    let mut shards: Vec<_> = total_stats.iter().collect();
    shards.sort_by_key(|(k, _)| *k);
    
    for (shard, count) in shards {
        let marker = if MONSTER_PRIMES.contains(shard) { "★" } else { " " };
        println!("  Shard {:2} {}: {:4} objects", shard, marker, count);
    }
    
    let total: usize = total_stats.values().sum();
    println!("\nTotal: {} objects across {} shards", total, total_stats.len());
    println!("\n✅ LMFDB sorted by Monster prime resonance!");
    
    Ok(())
}

fn process_lmfdb_file(path: &Path, output_dir: &Path) -> Result<HashMap<u32, usize>> {
    println!("Processing {:?}...", path.file_name().unwrap());
    
    let content = fs::read_to_string(path)?;
    let data: Value = serde_json::from_str(&content)?;
    
    let mut shards: HashMap<u32, Vec<Value>> = HashMap::new();
    
    // Sort objects into shards
    match data {
        Value::Array(objects) => {
            for obj in objects {
                let shard = assign_to_shard(&obj);
                shards.entry(shard).or_default().push(obj);
            }
        }
        Value::Object(map) => {
            for (key, obj) in map {
                let shard = assign_to_shard(&obj);
                let mut obj_with_id = obj.clone();
                if let Value::Object(ref mut m) = obj_with_id {
                    m.insert("_id".to_string(), Value::String(key));
                }
                shards.entry(shard).or_default().push(obj_with_id);
            }
        }
        _ => {}
    }
    
    // Write to shard files
    let filename = path.file_stem().unwrap().to_str().unwrap();
    let mut counts = HashMap::new();
    
    for (shard_num, objects) in &shards {
        let shard_dir = output_dir
            .join(format!("shard-{:02}", shard_num))
            .join("data")
            .join("lmfdb");
        
        fs::create_dir_all(&shard_dir)?;
        
        let output_file = shard_dir.join(format!("{}.json", filename));
        let json = serde_json::to_string_pretty(objects)?;
        fs::write(&output_file, json)?;
        
        counts.insert(*shard_num, objects.len());
        println!("  Shard {}: {} objects", shard_num, objects.len());
    }
    
    Ok(counts)
}

fn assign_to_shard(obj: &Value) -> u32 {
    let resonances = calculate_prime_resonance(obj);
    
    if resonances.is_empty() {
        return 1; // Default to shard 1
    }
    
    // Return prime with highest resonance
    resonances.iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(prime, _)| *prime)
        .unwrap_or(1)
}

fn calculate_prime_resonance(obj: &Value) -> Vec<(u32, f64)> {
    let numbers = extract_numbers(obj);
    
    if numbers.is_empty() {
        return vec![];
    }
    
    let mut resonances = Vec::new();
    
    for &prime in &MONSTER_PRIMES {
        let score: usize = numbers.iter()
            .filter(|&&n| n % prime == 0)
            .count();
        
        if score > 0 {
            let rate = score as f64 / numbers.len() as f64;
            resonances.push((prime, rate));
        }
    }
    
    resonances
}

fn extract_numbers(obj: &Value) -> Vec<u32> {
    let mut numbers = Vec::new();
    extract_numbers_recursive(obj, &mut numbers);
    numbers
}

fn extract_numbers_recursive(obj: &Value, numbers: &mut Vec<u32>) {
    match obj {
        Value::Number(n) => {
            if let Some(i) = n.as_u64() {
                if i > 0 && i <= u32::MAX as u64 {
                    numbers.push(i as u32);
                }
            }
        }
        Value::Array(arr) => {
            for item in arr {
                extract_numbers_recursive(item, numbers);
            }
        }
        Value::Object(map) => {
            for (_, value) in map {
                extract_numbers_recursive(value, numbers);
            }
        }
        _ => {}
    }
}
