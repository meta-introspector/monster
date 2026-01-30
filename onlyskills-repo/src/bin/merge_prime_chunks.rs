// Merge chunked prime data into Parquet
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct PrimeEntry {
    p: u64,
    g: i32,
    m: bool,
    s: u8,
}

#[derive(Debug, Deserialize)]
struct Chunk {
    chunk: u32,
    start: u64,
    end: u64,
    url: String,
    chord: Vec<u8>,
    primes: Vec<PrimeEntry>,
}

fn main() {
    println!("📦 Merging Prime Chunks to Parquet");
    println!("{}", "=".repeat(70));
    println!();
    
    let chunks_dir = Path::new("chunks");
    
    let mut all_primes = Vec::new();
    let mut all_chunks = Vec::new();
    let mut all_urls = Vec::new();
    let mut all_chords = Vec::new();
    
    // Read all chunk files
    let mut chunk_files: Vec<_> = fs::read_dir(chunks_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|s| s == "json").unwrap_or(false))
        .collect();
    
    chunk_files.sort_by_key(|e| e.path());
    
    println!("📖 Reading {} chunks...", chunk_files.len());
    
    for entry in chunk_files {
        let path = entry.path();
        let json = fs::read_to_string(&path).unwrap();
        
        if let Ok(chunk) = serde_json::from_str::<Chunk>(&json) {
            println!("  Chunk {}: {} primes, chord {:?}, {}", 
                chunk.chunk, chunk.primes.len(), chunk.chord, chunk.url);
            
            for prime in &chunk.primes {
                all_primes.push(prime.p);
                all_chunks.push(chunk.chunk);
                all_urls.push(chunk.url.clone());
                all_chords.push(format!("{},{},{}", chunk.chord[0], chunk.chord[1], chunk.chord[2]));
            }
        }
    }
    
    println!();
    println!("✓ Loaded {} total primes", all_primes.len());
    println!();
    
    // Create DataFrame
    let df = DataFrame::new(vec![
        Series::new("prime", all_primes),
        Series::new("chunk", all_chunks),
        Series::new("url", all_urls),
        Series::new("chord", all_chords),
    ]).unwrap();
    
    println!("📋 Sample data:");
    println!("{}", df.head(Some(10)));
    println!();
    
    // Write to Parquet
    let mut file = fs::File::create("prime_chunks.parquet").unwrap();
    ParquetWriter::new(&mut file).finish(&mut df.clone()).unwrap();
    
    println!("✓ Saved: prime_chunks.parquet");
    println!();
    println!("∞ All Chunks Merged. URLs. Chords. Parquet. ∞");
}
