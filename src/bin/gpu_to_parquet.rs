// Rust: GPU → Parquet pipeline (proper format)

use polars::prelude::*;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simulate GPU results (in real version, copy from CUDA)
    let qids: Vec<u64> = (0..1024).collect();
    let shards: Vec<u8> = qids.iter().map(|q| (q % 15) as u8).collect();
    let embeddings: Vec<f32> = (0..1024 * 4096).map(|i| i as f32).collect();
    
    // Create DataFrame
    let df = DataFrame::new(vec![
        Series::new("qid", qids),
        Series::new("shard", shards),
    ])?;
    
    // Write to parquet
    let mut file = File::create("qid_embeddings.parquet")?;
    ParquetWriter::new(&mut file).finish(&mut df.clone())?;
    
    println!("✅ Wrote {} rows to qid_embeddings.parquet", df.height());
    println!("   Columns: {:?}", df.get_column_names());
    
    Ok(())
}
