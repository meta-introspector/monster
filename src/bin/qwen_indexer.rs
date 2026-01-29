// Rust: Index 8M files with Qwen - Remove token layer to parquet

use polars::prelude::*;
use std::path::PathBuf;
use tokio::fs;

/// Token layer data (to be removed from model)
#[derive(Debug, Clone)]
pub struct TokenLayer {
    pub file_id: usize,
    pub tokens: Vec<u32>,
    pub embeddings: Vec<f32>,
    pub shard_id: u8,
}

/// Qwen indexer
pub struct QwenIndexer {
    output_dir: PathBuf,
    shard_size: usize,
}

impl QwenIndexer {
    pub fn new(output_dir: PathBuf, shard_size: usize) -> Self {
        Self { output_dir, shard_size }
    }
    
    /// Extract token layer from file
    pub async fn extract_token_layer(&self, file_path: PathBuf, file_id: usize) -> Result<TokenLayer, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(&file_path).await?;
        
        // Simulate tokenization (would use Qwen tokenizer)
        let tokens: Vec<u32> = content.bytes().map(|b| b as u32).collect();
        
        // Simulate embeddings (would use Qwen embeddings)
        let embeddings: Vec<f32> = tokens.iter().map(|&t| t as f32 / 256.0).collect();
        
        // Assign to shard by prime resonance
        let shard_id = (file_id % 71) as u8;
        
        Ok(TokenLayer {
            file_id,
            tokens,
            embeddings,
            shard_id,
        })
    }
    
    /// Write token layers to parquet shard
    pub fn write_shard(&self, layers: Vec<TokenLayer>, shard_id: u8) -> Result<(), Box<dyn std::error::Error>> {
        // Flatten to DataFrame
        let file_ids: Vec<_> = layers.iter().map(|l| l.file_id as i64).collect();
        let shard_ids: Vec<_> = layers.iter().map(|l| l.shard_id as i64).collect();
        let token_counts: Vec<_> = layers.iter().map(|l| l.tokens.len() as i64).collect();
        
        let df = DataFrame::new(vec![
            Series::new("file_id", file_ids),
            Series::new("shard_id", shard_ids),
            Series::new("token_count", token_counts),
        ])?;
        
        // Write to parquet
        let path = self.output_dir.join(format!("shard_{:02}.parquet", shard_id));
        let mut file = std::fs::File::create(path)?;
        ParquetWriter::new(&mut file).finish(&mut df.clone())?;
        
        Ok(())
    }
    
    /// Index all 8M files
    pub async fn index_all(&self, file_paths: Vec<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
        println!("📚 Indexing {} files with Qwen", file_paths.len());
        println!("="*70);
        println!();
        
        let mut shard_buffers: Vec<Vec<TokenLayer>> = vec![Vec::new(); 71];
        
        for (i, path) in file_paths.iter().enumerate() {
            // Extract token layer
            let layer = self.extract_token_layer(path.clone(), i).await?;
            let shard_id = layer.shard_id as usize;
            
            shard_buffers[shard_id].push(layer);
            
            // Flush shard when full
            if shard_buffers[shard_id].len() >= self.shard_size {
                println!("  Writing shard {} ({} files)", shard_id, shard_buffers[shard_id].len());
                self.write_shard(shard_buffers[shard_id].clone(), shard_id as u8)?;
                shard_buffers[shard_id].clear();
            }
            
            if (i + 1) % 100_000 == 0 {
                println!("  Processed {}/{} files", i + 1, file_paths.len());
            }
        }
        
        // Flush remaining
        for (shard_id, buffer) in shard_buffers.iter().enumerate() {
            if !buffer.is_empty() {
                println!("  Writing final shard {} ({} files)", shard_id, buffer.len());
                self.write_shard(buffer.clone(), shard_id as u8)?;
            }
        }
        
        println!();
        println!("✓ Indexed {} files into 71 shards", file_paths.len());
        
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Qwen Indexer - Remove Token Layer to Parquet");
    println!("="*70);
    println!();
    
    let output_dir = PathBuf::from("shards/qwen_tokens");
    std::fs::create_dir_all(&output_dir)?;
    
    let indexer = QwenIndexer::new(output_dir, 10_000);
    
    // Generate 8M file paths (example)
    let file_paths: Vec<PathBuf> = (0..8_000_000)
        .map(|i| PathBuf::from(format!("data/file_{}.txt", i)))
        .collect();
    
    println!("Configuration:");
    println!("  Files: {}", file_paths.len());
    println!("  Shards: 71 (Monster primes)");
    println!("  Shard size: 10,000 files");
    println!();
    
    indexer.index_all(file_paths).await?;
    
    println!();
    println!("="*70);
    println!("✅ Token layers removed and stored in parquet!");
    
    Ok(())
}
