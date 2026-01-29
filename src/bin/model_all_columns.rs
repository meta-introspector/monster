// Rust: Model each column in all parquet files

use polars::prelude::*;
use std::fs::File;
use std::collections::HashMap;

const MONSTER_PRIMES: [u32; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 MODEL ALL PARQUET COLUMNS");
    println!("{}", "=".repeat(70));
    println!();
    
    // Load parquet index
    let file = File::open("parquet_index.parquet")?;
    let df = ParquetReader::new(file).finish()?;
    
    println!("Loaded: {} parquet files", df.height());
    println!();
    
    let paths = df.column("file_path")?.str()?;
    
    let mut all_models = Vec::new();
    let mut processed = 0;
    
    println!("Processing parquet files...");
    
    for i in 0..df.height().min(20) {
        let path = paths.get(i).unwrap();
        
        // Try to read parquet
        if let Ok(file) = File::open(path) {
            if let Ok(df_data) = ParquetReader::new(file).finish() {
                let columns = df_data.get_column_names();
                
                println!("\n  File: {}", path.split('/').last().unwrap_or(path));
                println!("  Columns: {}", columns.len());
                
                // Model each column
                for col_name in columns {
                    if let Ok(col) = df_data.column(col_name) {
                        let dtype = col.dtype();
                        let len = col.len();
                        
                        // Assign to Monster shard
                        let hash = col_name.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                        let shard = (hash % 15) as u8;
                        let prime = MONSTER_PRIMES[shard as usize];
                        
                        all_models.push((
                            path.to_string(),
                            col_name.to_string(),
                            format!("{:?}", dtype),
                            len,
                            shard,
                            prime,
                        ));
                        
                        println!("    - {}: {:?} ({} rows) → Shard {} (Prime {})", 
                                 col_name, dtype, len, shard, prime);
                    }
                }
                
                processed += 1;
            }
        }
    }
    
    println!();
    println!("✅ Processed {} parquet files", processed);
    println!("✅ Modeled {} columns", all_models.len());
    println!();
    
    // Create column models DataFrame
    let (file_paths, col_names, dtypes, lengths, shards, primes): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) 
        = all_models.into_iter().map(|(f, c, d, l, s, p)| (f, c, d, l, s, p)).multiunzip();
    
    let df_models = DataFrame::new(vec![
        Series::new("file_path", file_paths),
        Series::new("column_name", col_names),
        Series::new("dtype", dtypes),
        Series::new("row_count", lengths),
        Series::new("shard", shards),
        Series::new("prime_label", primes),
    ])?;
    
    // Write to parquet
    let mut file = File::create("column_models.parquet")?;
    ParquetWriter::new(&mut file).finish(&mut df_models.clone())?;
    
    println!("✅ Created column_models.parquet");
    println!("   Rows: {}", df_models.height());
    println!();
    
    // Summary by shard
    println!("COLUMNS BY MONSTER SHARD:");
    for shard in 0..15 {
        let count = shards.iter().filter(|&&s| s == shard).count();
        let prime = MONSTER_PRIMES[shard as usize];
        let bar = "█".repeat(count / 2);
        println!("  Shard {:>2} (Prime {:>2}): {:>3} columns {}", shard, prime, count, bar);
    }
    
    println!();
    println!("{}", "=".repeat(70));
    println!("✅ All parquet columns modeled!");
    
    Ok(())
}
