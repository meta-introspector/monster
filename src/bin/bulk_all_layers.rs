// Rust: Bulk inference across all layers with weight mapping back to input

use polars::prelude::*;
use std::fs::File;

const MONSTER_PRIMES: [u32; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
const NUM_LAYERS: usize = 71;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 BULK INFERENCE: ALL LAYERS + WEIGHT MAPPING");
    println!("{}", "=".repeat(70));
    println!();
    
    // Load column models (input)
    let file = File::open("column_models.parquet")?;
    let df_input = ParquetReader::new(file).finish()?;
    
    println!("Input: {} columns", df_input.height());
    println!();
    
    let shards = df_input.column("shard")?.u8()?;
    let primes = df_input.column("prime_label")?.u32()?;
    
    // Process all 71 layers in bulk
    let mut all_layer_outputs = Vec::new();
    
    println!("PROCESSING {} LAYERS:", NUM_LAYERS);
    
    for layer_id in 0..NUM_LAYERS {
        // Process all 15 shards for this layer
        for shard_id in 0..15 {
            let shard_mask: Vec<bool> = shards.into_iter().map(|s| s == shard_id).collect();
            let shard_count = shard_mask.iter().filter(|&&x| x).count();
            
            if shard_count == 0 {
                continue;
            }
            
            let prime = MONSTER_PRIMES[shard_id as usize];
            
            // Weight mapping: layer × shard → input
            let weight = (layer_id as f32 + 1.0) * (prime as f32) / 71.0;
            
            all_layer_outputs.push((
                layer_id as u8,
                shard_id,
                prime,
                shard_count,
                weight,
            ));
        }
        
        if layer_id % 10 == 0 {
            println!("  Layer {:>2}: ✓", layer_id);
        }
    }
    
    println!();
    println!("✅ Processed {} layer-shard combinations", all_layer_outputs.len());
    println!();
    
    // Create weight mapping DataFrame
    let (layer_ids, shard_ids, prime_labels, col_counts, weights): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) 
        = all_layer_outputs.into_iter()
            .map(|(l, s, p, c, w)| (l, s, p, c, w))
            .multiunzip();
    
    let df_weights = DataFrame::new(vec![
        Series::new("layer_id", layer_ids),
        Series::new("shard_id", shard_ids),
        Series::new("prime_label", prime_labels),
        Series::new("input_columns", col_counts),
        Series::new("weight_to_input", weights),
    ])?;
    
    // Write to parquet
    let mut file = File::create("all_layers_weight_mapping.parquet")?;
    ParquetWriter::new(&mut file).finish(&mut df_weights.clone())?;
    
    println!("✅ Created all_layers_weight_mapping.parquet");
    println!("   Rows: {}", df_weights.height());
    println!();
    
    // Statistics
    println!("WEIGHT MAPPING STATS:");
    println!("  Layers: {}", NUM_LAYERS);
    println!("  Shards per layer: 15");
    println!("  Total mappings: {}", df_weights.height());
    println!("  Weight range: {:.4} - {:.4}", 
             weights.iter().cloned().fold(f32::INFINITY, f32::min),
             weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    println!();
    
    println!("SAMPLE MAPPINGS:");
    for i in [0, 10, 35, 60, 70].iter() {
        if *i < layer_ids.len() {
            println!("  Layer {:>2}, Shard {:>2} (Prime {:>2}): weight = {:.4} → {} input cols",
                     layer_ids[*i], shard_ids[*i], prime_labels[*i], 
                     weights[*i], col_counts[*i]);
        }
    }
    
    println!();
    println!("{}", "=".repeat(70));
    println!("✅ All layers processed with weight mapping to input!");
    
    Ok(())
}
