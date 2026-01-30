// Rust version of shard_lmfdb_by_71.py
// Shard entire LMFDB into 71 shards using hash % 71

use std::fs;
use std::collections::HashMap;
use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Chunk {
    file: String,
    code: String,
    bytes: usize,
    shard: u8,
    hash: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ShardStats {
    shard_id: u8,
    chunk_count: usize,
    total_bytes: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔮 SHARDING LMFDB INTO 71 SHARDS");
    println!("{}", "=".repeat(70));
    println!();
    
    // Load chunks
    let json = fs::read_to_string("lmfdb_71_chunks.json")?;
    let mut chunks: Vec<Chunk> = serde_json::from_str(&json)?;
    
    println!("Loaded {} chunks", chunks.len());
    println!();
    
    // Shard by hash % 71
    let mut shards: HashMap<u8, Vec<Chunk>> = HashMap::new();
    
    for chunk in &mut chunks {
        if !chunk.code.is_empty() {
            let mut hasher = Sha256::new();
            hasher.update(&chunk.code);
            let hash = hasher.finalize();
            
            let shard_id = (u64::from_be_bytes(hash[0..8].try_into()?) % 71) as u8;
            chunk.shard = shard_id;
            chunk.hash = format!("{:x}", hash)[..16].to_string();
            
            shards.entry(shard_id).or_default().push(chunk.clone());
        }
    }
    
    println!("📊 SHARD DISTRIBUTION:");
    println!("{}", "-".repeat(70));
    
    let mut stats = Vec::new();
    
    for shard_id in 0..71 {
        if let Some(shard_chunks) = shards.get(&shard_id) {
            let count = shard_chunks.len();
            let total_bytes: usize = shard_chunks.iter().map(|c| c.bytes).sum();
            
            println!("Shard {:2}: {:3} chunks, {:8} bytes", shard_id, count, total_bytes);
            
            stats.push(ShardStats {
                shard_id,
                chunk_count: count,
                total_bytes,
            });
            
            // Save shard
            let shard_json = serde_json::to_string_pretty(&shard_chunks)?;
            fs::write(format!("lmfdb_shard_{:02}.json", shard_id), shard_json)?;
        }
    }
    
    // Save stats
    let stats_json = serde_json::to_string_pretty(&stats)?;
    fs::write("shard_stats.json", stats_json)?;
    
    println!();
    println!("✅ Created 71 shards");
    println!("💾 Saved to: lmfdb_shard_XX.json");
    
    Ok(())
}
