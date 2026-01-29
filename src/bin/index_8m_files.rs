// Rust: Index 8M files into parquet shards

use polars::prelude::*;
use std::fs::{self, File};
use std::path::Path;
use walkdir::WalkDir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📇 INDEXING 8M FILES → PARQUET SHARDS");
    println!("{}", "=".repeat(70));
    println!();
    
    let mut file_paths = Vec::new();
    let mut file_sizes = Vec::new();
    let mut file_types = Vec::new();
    let mut shards = Vec::new();
    let mut layers = Vec::new();
    
    println!("Scanning files...");
    
    let mut count = 0;
    for entry in WalkDir::new(".").max_depth(10).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            if let Ok(metadata) = entry.metadata() {
                let path = entry.path().to_string_lossy().to_string();
                let size = metadata.len();
                let ext = Path::new(&path)
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                
                file_paths.push(path);
                file_sizes.push(size);
                file_types.push(ext);
                shards.push((count % 15) as u8);
                layers.push((count % 71) as u8);
                
                count += 1;
                
                if count % 100000 == 0 {
                    println!("  Indexed: {}", count);
                }
            }
        }
    }
    
    println!();
    println!("✅ Indexed {} files", count);
    println!();
    
    // Create DataFrame
    let df = DataFrame::new(vec![
        Series::new("file_path", file_paths),
        Series::new("file_size", file_sizes),
        Series::new("file_type", file_types),
        Series::new("shard", shards),
        Series::new("layer", layers),
    ])?;
    
    // Write to parquet
    let mut file = File::create("file_index_8m.parquet")?;
    ParquetWriter::new(&mut file).finish(&mut df.clone())?;
    
    println!("✅ Created file_index_8m.parquet");
    println!("   Rows: {}", df.height());
    println!("   Shards: 15");
    println!("   Layers: 71");
    println!();
    println!("{}", "=".repeat(70));
    println!("✅ 8M files indexed!");
    
    Ok(())
}
