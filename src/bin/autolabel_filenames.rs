// Rust: Auto-label filename tokens with prime harmonics

use polars::prelude::*;
use std::fs::File;
use std::collections::HashMap;

const MONSTER_PRIMES: [u32; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
const BASE_FREQ: f64 = 440.0;

fn prime_to_frequency(prime: u32) -> f64 {
    BASE_FREQ * 1.0594630943592953_f64.powi(prime as i32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 AUTO-LABELING FILENAME TOKENS WITH PRIME HARMONICS");
    println!("{}", "=".repeat(70));
    println!();
    
    // Calculate prime harmonics
    let prime_freqs: HashMap<u32, f64> = MONSTER_PRIMES.iter()
        .map(|&p| (p, prime_to_frequency(p)))
        .collect();
    
    println!("PRIME HARMONICS:");
    for (i, &prime) in MONSTER_PRIMES.iter().take(5).enumerate() {
        println!("  Prime {:>2} → {:>8.2} Hz", prime, prime_freqs[&prime]);
    }
    println!("  ...");
    println!();
    
    // Load file mappings
    let file = File::open("files_to_layer1_shards.parquet")?;
    let df = ParquetReader::new(file).finish()?;
    
    println!("Loaded: {} files", df.height());
    println!();
    
    // Extract data
    let paths = df.column("file_path")?.str()?;
    let shards = df.column("layer1_shard")?.u8()?;
    
    let mut file_paths = Vec::new();
    let mut token_bits = Vec::new();
    let mut prime_labels = Vec::new();
    let mut harmonic_freqs = Vec::new();
    let mut shard_ids = Vec::new();
    
    println!("AUTO-LABELING:");
    
    for i in 0..df.height().min(1000) {
        let path = paths.get(i).unwrap();
        let shard = shards.get(i).unwrap();
        
        // Extract filename
        let filename = path.split('/').last().unwrap_or(path);
        
        // Convert to bits
        let hash = filename.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let bits = format!("{:064b}", hash);
        
        // Label with prime harmonic
        let prime = MONSTER_PRIMES[shard as usize];
        let freq = prime_freqs[&prime];
        
        if i < 5 {
            println!("  {:30} → Prime {:>2} → {:>8.2} Hz", 
                     &filename[..filename.len().min(30)], prime, freq);
        }
        
        file_paths.push(path.to_string());
        token_bits.push(bits);
        prime_labels.push(prime);
        harmonic_freqs.push(freq);
        shard_ids.push(shard);
    }
    
    println!();
    println!("✅ Auto-labeled {} filenames", file_paths.len());
    println!();
    
    // Create labeled dataset
    let df_labeled = DataFrame::new(vec![
        Series::new("file_path", file_paths),
        Series::new("token_bits", token_bits),
        Series::new("prime_label", prime_labels),
        Series::new("harmonic_freq", harmonic_freqs),
        Series::new("shard", shard_ids),
    ])?;
    
    // Write to parquet
    let mut file = File::create("filename_tokens_labeled.parquet")?;
    ParquetWriter::new(&mut file).finish(&mut df_labeled.clone())?;
    
    println!("✅ Created filename_tokens_labeled.parquet");
    println!("   Rows: {}", df_labeled.height());
    println!("   Columns: {:?}", df_labeled.get_column_names());
    println!();
    
    // Label distribution
    println!("PRIME LABEL DISTRIBUTION:");
    for &prime in MONSTER_PRIMES.iter().take(10) {
        let count = prime_labels.iter().filter(|&&p| p == prime).count();
        let freq = prime_freqs[&prime];
        let bar = "█".repeat(count / 10);
        println!("  Prime {:>2} ({:>8.2} Hz): {:>3} files {}", prime, freq, count, bar);
    }
    
    println!();
    println!("{}", "=".repeat(70));
    println!("✅ Filenames auto-labeled with prime harmonics!");
    
    Ok(())
}
