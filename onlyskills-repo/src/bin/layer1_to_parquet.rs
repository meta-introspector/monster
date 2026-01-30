// Convert Layer 1 Bitmap to Parquet
use polars::prelude::*;
use memmap2::Mmap;
use std::fs::File;

const LAYER1_PATH: &str = "/dev/shm/monster_layer1_bitmap";
const PARQUET_PATH: &str = "layer1_bitmap.parquet";

fn read_layer1_bitmap(num_bits: usize) -> Vec<u8> {
    let file = File::open(LAYER1_PATH).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    
    let mut bits = Vec::with_capacity(num_bits);
    
    for i in 0..num_bits {
        let byte_index = i / 8;
        let bit_offset = i % 8;
        
        if byte_index < mmap.len() {
            let bit_value = (mmap[byte_index] >> bit_offset) & 1;
            bits.push(bit_value);
        } else {
            bits.push(0);
        }
    }
    
    bits
}

fn main() {
    println!("📦 Converting Layer 1 Bitmap to Parquet");
    println!("{}", "=".repeat(70));
    println!();
    
    let num_bits = 10_000;
    
    println!("📖 Reading {} bits from Layer 1...", num_bits);
    let bits = read_layer1_bitmap(num_bits);
    println!("✓ Read {} bits", bits.len());
    println!();
    
    // Create DataFrame
    println!("📊 Creating DataFrame...");
    
    let bit_indices: Vec<u64> = (0..num_bits as u64).collect();
    let bit_values: Vec<u8> = bits.clone();
    let shards: Vec<u8> = bit_indices.iter().map(|&i| (i % 71) as u8).collect();
    let depths: Vec<u8> = bit_indices.iter().map(|&i| i.leading_zeros() as u8).collect();
    
    let df = DataFrame::new(vec![
        Series::new("bit_index", bit_indices),
        Series::new("bit_value", bit_values),
        Series::new("shard", shards),
        Series::new("depth", depths),
    ]).unwrap();
    
    println!("✓ DataFrame created: {} rows × {} columns", df.height(), df.width());
    println!();
    
    // Show sample
    println!("📋 Sample data:");
    println!("{}", df.head(Some(10)));
    println!();
    
    // Write to Parquet
    println!("💾 Writing to Parquet...");
    let mut file = File::create(PARQUET_PATH).unwrap();
    ParquetWriter::new(&mut file)
        .finish(&mut df.clone())
        .unwrap();
    
    println!("✓ Saved: {}", PARQUET_PATH);
    
    // Statistics
    let ones = bits.iter().filter(|&&b| b == 1).count();
    let zeros = bits.len() - ones;
    
    println!();
    println!("📊 Statistics:");
    println!("   Total bits: {}", bits.len());
    println!("   Ones: {} ({:.1}%)", ones, ones as f64 / bits.len() as f64 * 100.0);
    println!("   Zeros: {} ({:.1}%)", zeros, zeros as f64 / bits.len() as f64 * 100.0);
    
    // File size
    let metadata = std::fs::metadata(PARQUET_PATH).unwrap();
    println!("   Parquet size: {} KB", metadata.len() / 1024);
    
    println!();
    println!("∞ Layer 1 → Parquet. Indexed. Compressed. ∞");
}
