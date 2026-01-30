// Hierarchical Markov model: column → shard → bitwise sampling
// Based on crossbeam best practices

use polars::prelude::*;
use std::collections::HashMap;
use std::fs::{self, File};
use std::sync::{Arc, Mutex};
use crossbeam::channel::{bounded, Receiver, Sender};
use std::thread;
use serde::{Serialize, Deserialize};

const MONSTER_PRIMES: [u32; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
const NUM_WORKERS: usize = 24;

#[derive(Clone, Serialize, Deserialize)]
struct ColumnMarkov {
    file: String,
    column: String,
    shard: u8,
    layer: u8,
    transitions: HashMap<u8, HashMap<u8, u32>>, // byte -> next_byte -> count
    total_bytes: usize,
}

#[derive(Serialize, Deserialize)]
struct ShardMarkov {
    shard: u8,
    layer: u8,
    merged_transitions: HashMap<u8, HashMap<u8, f64>>, // byte -> next_byte -> probability
    total_columns: usize,
}

fn compute_column_markov(
    file: &str,
    col_name: &str,
    col: &Column,
    shard: u8,
    layer: u8,
) -> Option<ColumnMarkov> {
    let mut transitions: HashMap<u8, HashMap<u8, u32>> = HashMap::new();
    let mut total_bytes = 0;
    
    for row in 0..col.len().min(1000) {
        let value = match col.get(row) {
            Ok(v) => format!("{:?}", v),
            Err(_) => continue,
        };
        
        let bytes: Vec<u8> = value.bytes().collect();
        for window in bytes.windows(2) {
            let curr = window[0];
            let next = window[1];
            
            *transitions.entry(curr).or_default().entry(next).or_default() += 1;
            total_bytes += 1;
        }
    }
    
    if total_bytes == 0 {
        return None;
    }
    
    Some(ColumnMarkov {
        file: file.to_string(),
        column: col_name.to_string(),
        shard,
        layer,
        transitions,
        total_bytes,
    })
}

fn worker(
    worker_id: usize,
    rx: Receiver<String>,
    tx: Sender<Vec<ColumnMarkov>>,
    skipped: Arc<Mutex<Vec<String>>>,
) {
    while let Ok(parquet_path) = rx.recv() {
        match process_parquet(&parquet_path) {
            Ok(models) => {
                if !models.is_empty() {
                    let _ = tx.send(models);
                }
            }
            Err(e) => {
                if worker_id == 0 {
                    eprintln!("Worker {}: Error {}: {}", worker_id, parquet_path, e);
                }
                if let Ok(mut skip) = skipped.lock() {
                    skip.push(parquet_path);
                }
            }
        }
    }
}

fn process_parquet(path: &str) -> Result<Vec<ColumnMarkov>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let df = ParquetReader::new(file).finish()?;
    let mut models = Vec::new();
    
    for col_name in df.get_column_names() {
        let col = match df.column(col_name) {
            Ok(c) => c,
            Err(_) => continue,
        };
        
        // Skip unsupported types
        let dtype_str = format!("{:?}", col.dtype());
        if dtype_str.contains("UInt8") || dtype_str.contains("UInt16") || 
           dtype_str.contains("Struct") || dtype_str.contains("List") {
            continue;
        }
        
        let hash = col_name.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let shard = (hash % 15) as u8;
        
        for layer in 0..71 {
            if let Some(model) = compute_column_markov(path, col_name, col, shard, layer) {
                models.push(model);
            }
        }
    }
    
    Ok(models)
}

fn merge_shard_models(column_models: &[ColumnMarkov]) -> Vec<ShardMarkov> {
    let mut shard_map: HashMap<(u8, u8), Vec<&ColumnMarkov>> = HashMap::new();
    
    for model in column_models {
        shard_map.entry((model.shard, model.layer)).or_default().push(model);
    }
    
    let mut shard_models = Vec::new();
    
    for ((shard, layer), models) in shard_map {
        let mut merged: HashMap<u8, HashMap<u8, u32>> = HashMap::new();
        
        for model in &models {
            for (&curr, nexts) in &model.transitions {
                for (&next, &count) in nexts {
                    *merged.entry(curr).or_default().entry(next).or_default() += count;
                }
            }
        }
        
        // Convert counts to probabilities
        let mut merged_probs: HashMap<u8, HashMap<u8, f64>> = HashMap::new();
        for (curr, nexts) in merged {
            let total: u32 = nexts.values().sum();
            let probs: HashMap<u8, f64> = nexts.iter()
                .map(|(&next, &count)| (next, count as f64 / total as f64))
                .collect();
            merged_probs.insert(curr, probs);
        }
        
        shard_models.push(ShardMarkov {
            shard,
            layer,
            merged_transitions: merged_probs,
            total_columns: models.len(),
        });
    }
    
    shard_models
}

fn sample_bitwise(model: &ShardMarkov, num_bytes: usize, seed: u8) -> Vec<u8> {
    let mut result = Vec::with_capacity(num_bytes);
    let mut current = seed;
    
    for _ in 0..num_bytes {
        result.push(current);
        
        if let Some(nexts) = model.merged_transitions.get(&current) {
            // Sample next byte based on probabilities
            let rng = (current as u32).wrapping_mul(31).wrapping_add(result.len() as u32);
            let rand = (rng % 1000) as f64 / 1000.0;
            
            let mut cumulative = 0.0;
            let mut next_byte = current;
            
            for (&byte, &prob) in nexts {
                cumulative += prob;
                if rand < cumulative {
                    next_byte = byte;
                    break;
                }
            }
            
            current = next_byte;
        } else {
            // No transitions, use hash
            current = current.wrapping_mul(31).wrapping_add(1);
        }
    }
    
    result
}

fn writer_thread(
    rx: Receiver<Vec<ColumnMarkov>>,
) -> thread::JoinHandle<Result<Vec<ColumnMarkov>, String>> {
    thread::spawn(move || {
        let mut all_models = Vec::new();
        let mut batches = 0;
        
        while let Ok(models) = rx.recv() {
            all_models.extend(models);
            batches += 1;
            
            if batches % 100 == 0 {
                println!("  Collected {} batches, {} column models", batches, all_models.len());
            }
        }
        
        Ok(all_models)
    })
}

fn find_parquet_files(file_list: Option<&str>) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    if let Some(path) = file_list {
        let content = fs::read_to_string(path)?;
        Ok(content.lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.ends_with(".parquet"))
            .collect())
    } else {
        Ok(Vec::new())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 MARKOV MODEL: COLUMN → SHARD → BITWISE SAMPLING");
    println!("{}", "=".repeat(70));
    println!();
    
    let file_list = std::env::args().nth(1);
    let parquet_files = find_parquet_files(file_list.as_deref())?;
    println!("Found {} parquet files", parquet_files.len());
    println!();
    
    // Channels
    let (work_tx, work_rx) = bounded::<String>(1000);
    let (result_tx, result_rx) = bounded::<Vec<ColumnMarkov>>(100);
    let skipped = Arc::new(Mutex::new(Vec::new()));
    
    // Spawn workers
    for worker_id in 0..NUM_WORKERS {
        let rx = work_rx.clone();
        let tx = result_tx.clone();
        let skip = Arc::clone(&skipped);
        
        thread::spawn(move || {
            worker(worker_id, rx, tx, skip);
        });
    }
    
    drop(work_rx);
    drop(result_tx);
    
    // Spawn writer
    let writer = writer_thread(result_rx);
    
    // Queue work
    println!("🚀 Computing column Markov models...");
    for file in parquet_files.iter() {
        work_tx.send(file.clone())?;
    }
    drop(work_tx);
    
    println!("⏳ Processing...");
    
    // Wait for writer
    let column_models = writer.join().unwrap()?;
    
    println!();
    println!("✅ Computed {} column Markov models", column_models.len());
    println!();
    
    // Merge into shard models
    println!("🔀 Merging into shard models...");
    let shard_models = merge_shard_models(&column_models);
    println!("✅ Created {} shard models", shard_models.len());
    println!();
    
    // Sample bitwise from each shard
    println!("🎲 Sampling bitwise from shards...");
    fs::create_dir_all("markov_samples")?;
    
    for model in &shard_models {
        let sample = sample_bitwise(model, 1024, model.shard);
        let filename = format!("markov_samples/shard_{:02}_layer_{:02}.bin", model.shard, model.layer);
        fs::write(&filename, &sample)?;
        
        if model.layer % 10 == 0 {
            println!("  Shard {} Layer {}: {} columns, {} bytes sampled", 
                model.shard, model.layer, model.total_columns, sample.len());
        }
    }
    
    // Save shard models
    let json = serde_json::to_string_pretty(&shard_models)?;
    fs::write("markov_shard_models.json", json)?;
    
    println!();
    println!("{}", "=".repeat(70));
    println!("✅ Markov models complete");
    println!("📊 Column models: {}", column_models.len());
    println!("📊 Shard models: {}", shard_models.len());
    println!("💾 Samples saved to: markov_samples/");
    println!("💾 Models saved to: markov_shard_models.json");
    
    Ok(())
}
