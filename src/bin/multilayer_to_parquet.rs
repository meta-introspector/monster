// Rust: Multi-layer mining → Parquet

use polars::prelude::*;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📦 MULTI-LAYER MINING → PARQUET");
    println!("{}", "=".repeat(70));
    println!();
    
    let num_layers = 71;
    let batch_size = 1024;
    
    // Generate data for all layers
    let mut layer_ids = Vec::new();
    let mut frequencies = Vec::new();
    let mut tokens = Vec::new();
    let mut shards = Vec::new();
    
    for layer in 0..num_layers {
        for i in 0..batch_size {
            layer_ids.push(layer);
            frequencies.push(440.0 * (layer + 1) as f32);
            tokens.push((i * layer) % 50000);
            shards.push((i % 15) as u8);
        }
    }
    
    println!("Generated {} points across {} layers", layer_ids.len(), num_layers);
    
    // Create DataFrame
    let df = DataFrame::new(vec![
        Series::new("layer", layer_ids),
        Series::new("frequency", frequencies),
        Series::new("token", tokens),
        Series::new("shard", shards),
    ])?;
    
    // Write to parquet
    let mut file = File::create("multilayer_lattice.parquet")?;
    ParquetWriter::new(&mut file).finish(&mut df.clone())?;
    
    println!("✅ Wrote {} rows to multilayer_lattice.parquet", df.height());
    println!("   Columns: {:?}", df.get_column_names());
    println!();
    println!("{}", "=".repeat(70));
    println!("✅ Multi-layer lattice captured!");
    
    Ok(())
}
