// Vectorize ALL parquets with crossbeam best practices (24 CPUs)
// Merged: v1 + v2 + semantic analysis

use polars::prelude::*;
use std::fs::{self, File};
use std::sync::{Arc, Mutex};
use crossbeam::channel::{bounded, Receiver, Sender};
use std::thread;
use serde::{Serialize, Deserialize};

const MONSTER_PRIMES: [u32; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
const NUM_WORKERS: usize = 24;
const BATCH_SIZE: usize = 10000;

#[derive(Clone)]
struct CellVector {
    file: String,
    column: String,
    row: usize,
    value_hash: u64,
    shard: u8,
    layer: u8,
}

#[derive(Serialize, Deserialize)]
struct AnalysisResults {
    total_files: usize,
    total_vectors: usize,
    shards_written: usize,
    skipped_files: Vec<String>,
}

fn vectorize_cell(value: &str, shard: u8, layer: u8) -> Vec<f32> {
    let hash = value.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let prime = MONSTER_PRIMES[shard as usize] as f32;
    let layer_weight = (layer as f32 + 1.0) * prime / 71.0;
    
    (0..4096).map(|i| {
        ((hash.wrapping_add(i as u64) as f32) * layer_weight).sin()
    }).collect()
}

fn worker(
    worker_id: usize,
    rx: Receiver<String>,
    tx: Sender<Vec<CellVector>>,
    skipped: Arc<Mutex<Vec<String>>>,
) {
    while let Ok(parquet_path) = rx.recv() {
        match process_parquet(&parquet_path) {
            Ok(vectors) => {
                if !vectors.is_empty() {
                    let _ = tx.send(vectors);
                }
            }
            Err(e) => {
                eprintln!("Worker {}: Error processing {}: {}", worker_id, parquet_path, e);
                if let Ok(mut skip) = skipped.lock() {
                    skip.push(parquet_path);
                }
            }
        }
    }
}

fn process_parquet(path: &str) -> Result<Vec<CellVector>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let df = ParquetReader::new(file).finish()?;
    let mut vectors = Vec::new();
    
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
        
        for row in 0..col.len().min(100) {
            let value = match col.get(row) {
                Ok(v) => format!("{:?}", v),
                Err(_) => continue,
            };
            let value_hash = value.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            
            for layer in 0..71 {
                vectors.push(CellVector {
                    file: path.to_string(),
                    column: col_name.to_string(),
                    row,
                    value_hash,
                    shard,
                    layer,
                });
            }
        }
    }
    
    Ok(vectors)
}

fn writer_thread(
    rx: Receiver<Vec<CellVector>>,
) -> thread::JoinHandle<Result<Vec<Vec<CellVector>>, String>> {
    thread::spawn(move || {
        let mut shard_vectors: Vec<Vec<CellVector>> = (0..71).map(|_| Vec::new()).collect();
        let mut batches_processed = 0;
        
        while let Ok(vectors) = rx.recv() {
            for v in vectors {
                shard_vectors[v.layer as usize].push(v);
            }
            batches_processed += 1;
            
            if batches_processed % 100 == 0 {
                let total: usize = shard_vectors.iter().map(|s| s.len()).sum();
                println!("  Processed: {} batches, {} vectors", batches_processed, total);
            }
        }
        
        Ok(shard_vectors)
    })
}

fn find_parquet_files(file_list: Option<&str>) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    if let Some(path) = file_list {
        // Read from .txt file
        let content = fs::read_to_string(path)?;
        Ok(content.lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.ends_with(".parquet"))
            .collect())
    } else {
        // Use locate command
        let output = std::process::Command::new("locate")
            .arg("*.parquet")
            .output()?;
        
        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|line| line.ends_with(".parquet"))
            .map(|s| s.to_string())
            .collect())
    }
}

fn write_shards(shard_vectors: Vec<Vec<CellVector>>) -> Result<usize, Box<dyn std::error::Error>> {
    let mut shards_written = 0;
    
    for (layer, vectors) in shard_vectors.iter().enumerate() {
        if vectors.is_empty() { continue; }
        
        println!("  💾 Writing shard {} ({} vectors)...", layer, vectors.len());
        
        let df = df! {
            "file" => vectors.iter().map(|v| v.file.clone()).collect::<Vec<_>>(),
            "column" => vectors.iter().map(|v| v.column.clone()).collect::<Vec<_>>(),
            "row" => vectors.iter().map(|v| v.row as u32).collect::<Vec<_>>(),
            "value_hash" => vectors.iter().map(|v| v.value_hash).collect::<Vec<_>>(),
            "shard" => vectors.iter().map(|v| v.shard as u32).collect::<Vec<_>>(),
            "layer" => vectors.iter().map(|v| v.layer as u32).collect::<Vec<_>>(),
        }?;
        
        let filename = format!("vectors_layer_{:02}.parquet", layer);
        let mut file = File::create(&filename)?;
        ParquetWriter::new(&mut file).finish(&mut df.clone())?;
        shards_written += 1;
    }
    
    Ok(shards_written)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 VECTORIZING ALL PARQUETS (24 CPUs)");
    println!("{}", "=".repeat(70));
    println!();
    
    // Check for file list argument
    let file_list = std::env::args().nth(1);
    
    // Find parquet files
    let parquet_files = find_parquet_files(file_list.as_deref())?;
    println!("Found {} parquet files", parquet_files.len());
    println!();
    
    // Channels with backpressure
    let (work_tx, work_rx) = bounded::<String>(1000);
    let (result_tx, result_rx) = bounded::<Vec<CellVector>>(100);
    
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
    println!("🚀 Queueing {} files...", parquet_files.len());
    for file in parquet_files.iter() {
        work_tx.send(file.clone())?;
    }
    drop(work_tx);
    
    println!("⏳ Processing...");
    
    // Wait for writer
    let shard_vectors = writer.join().unwrap()?;
    
    println!();
    println!("✅ All workers finished");
    
    let total_vectors: usize = shard_vectors.iter().map(|s| s.len()).sum();
    println!("Total vectors: {}", total_vectors);
    println!();
    
    // Write shards
    println!("💾 Writing shards...");
    let shards_written = write_shards(shard_vectors)?;
    
    // Save analysis
    let skipped_files = skipped.lock().unwrap().clone();
    let results = AnalysisResults {
        total_files: parquet_files.len(),
        total_vectors,
        shards_written,
        skipped_files: skipped_files.clone(),
    };
    
    let json = serde_json::to_string_pretty(&results)?;
    fs::write("vectorize_results.json", json)?;
    
    println!();
    println!("{}", "=".repeat(70));
    println!("✅ Created {} shard files", shards_written);
    println!("📊 Skipped {} files", skipped_files.len());
    println!("💾 Results saved to: vectorize_results.json");
    println!();
    println!("Next: Run analyze_semantic_significance to test statistical significance");
    
    Ok(())
}
