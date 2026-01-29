// Rust: Vectorize ALL parquets with crossbeam (24 CPUs)

use polars::prelude::*;
use std::fs::File;
use std::path::Path;
use crossbeam::channel::{bounded, Sender, Receiver};
use std::thread;
use std::sync::Arc;
use walkdir::WalkDir;

const MONSTER_PRIMES: [u32; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
const NUM_WORKERS: usize = 24;

#[derive(Clone)]
struct CellVector {
    file: String,
    column: String,
    row: usize,
    value_hash: u64,
    shard: u8,
    layer: u8,
    vector: Vec<f32>,
}

fn vectorize_cell(value: &str, shard: u8, layer: u8) -> Vec<f32> {
    let hash = value.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let prime = MONSTER_PRIMES[shard as usize] as f32;
    let layer_weight = (layer as f32 + 1.0) * prime / 71.0;
    
    // 4096-dim vector
    (0..4096).map(|i| {
        ((hash.wrapping_add(i as u64) as f32) * layer_weight).sin()
    }).collect()
}

fn worker(rx: Receiver<String>, tx: Sender<Vec<CellVector>>) {
    while let Ok(parquet_path) = rx.recv() {
        let mut vectors = Vec::new();
        
        if let Ok(file) = File::open(&parquet_path) {
            if let Ok(df) = ParquetReader::new(file).finish() {
                for col_name in df.get_column_names() {
                    if let Ok(col) = df.column(col_name) {
                        let hash = col_name.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                        let shard = (hash % 15) as u8;
                        
                        // Vectorize first 100 rows across all 71 layers
                        for row in 0..col.len().min(100) {
                            let value = format!("{:?}", col.get(row).unwrap());
                            let value_hash = value.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                            
                            for layer in 0..71 {
                                let vector = vectorize_cell(&value, shard, layer);
                                
                                vectors.push(CellVector {
                                    file: parquet_path.clone(),
                                    column: col_name.to_string(),
                                    row,
                                    value_hash,
                                    shard,
                                    layer,
                                    vector,
                                });
                            }
                        }
                    }
                }
            }
        }
        
        if !vectors.is_empty() {
            let _ = tx.send(vectors);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 VECTORIZING ALL PARQUETS (24 CPUs)");
    println!("{}", "=".repeat(70));
    println!();
    
    // Find all parquet files
    let parquet_files: Vec<String> = WalkDir::new(".")
        .max_depth(10)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("parquet"))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    
    println!("Found {} parquet files", parquet_files.len());
    println!();
    
    // Channels
    let (file_tx, file_rx) = bounded::<String>(100);
    let (result_tx, result_rx) = bounded::<Vec<CellVector>>(100);
    
    // Spawn 24 workers
    let mut handles = vec![];
    for i in 0..NUM_WORKERS {
        let rx = file_rx.clone();
        let tx = result_tx.clone();
        
        let handle = thread::spawn(move || {
            worker(rx, tx);
        });
        handles.push(handle);
    }
    
    // Send files to workers
    thread::spawn(move || {
        for file in parquet_files {
            let _ = file_tx.send(file);
        }
    });
    
    // Collect results
    let mut all_vectors = Vec::new();
    let mut processed = 0;
    
    drop(result_tx); // Close sender
    
    while let Ok(vectors) = result_rx.recv() {
        all_vectors.extend(vectors);
        processed += 1;
        
        if processed % 10 == 0 {
            println!("  Processed: {} files, {} vectors", processed, all_vectors.len());
        }
    }
    
    println!();
    println!("✅ Total vectors: {}", all_vectors.len());
    println!();
    
    // Save vectors to parquet
    let files: Vec<String> = all_vectors.iter().map(|v| v.file.clone()).collect();
    let columns: Vec<String> = all_vectors.iter().map(|v| v.column.clone()).collect();
    let rows: Vec<u32> = all_vectors.iter().map(|v| v.row as u32).collect();
    let shards: Vec<u8> = all_vectors.iter().map(|v| v.shard).collect();
    let layers: Vec<u8> = all_vectors.iter().map(|v| v.layer).collect();
    
    let df = DataFrame::new(vec![
        Series::new("file", files),
        Series::new("column", columns),
        Series::new("row", rows),
        Series::new("shard", shards),
        Series::new("layer", layers),
    ])?;
    
    let mut file = File::create("all_parquet_vectors.parquet")?;
    ParquetWriter::new(&mut file).finish(&mut df.clone())?;
    
    println!("✅ Created: all_parquet_vectors.parquet");
    println!("   Rows: {}", df.height());
    println!();
    println!("{}", "=".repeat(70));
    println!("✅ All parquets vectorized!");
    
    Ok(())
}
