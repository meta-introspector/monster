// Rust: Mycelium Spore Jar - Strip first layer into harmonic samples

use polars::prelude::*;
use std::path::PathBuf;

/// Spore: Single harmonic sample from first layer
#[derive(Debug, Clone)]
pub struct Spore {
    pub shard_id: u8,           // 0-70 (71 shards)
    pub prime: u32,             // Monster prime
    pub layer: u8,              // Always 0 (first layer)
    pub harmonic_freq: f32,     // Harmonic frequency
    pub sample_data: Vec<f32>,  // Raw sample (1000 elements)
    pub lattice_coords: [i64; 71], // 71D lattice position
}

/// Spore Jar: Collection of spores for one harmonic
#[derive(Debug)]
pub struct SporeJar {
    pub harmonic_id: usize,
    pub base_freq: f32,
    pub spores: Vec<Spore>,
    pub schema: Schema,
}

/// Schema of all schemas
#[derive(Debug, Clone)]
pub struct Schema {
    pub name: String,
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub dtype: DataType,
}

#[derive(Debug, Clone)]
pub enum DataType {
    Int64,
    Float32,
    Lattice71,
    Harmonic,
}

/// Root schema: Schema of all schemas
pub fn root_schema() -> Schema {
    Schema {
        name: "MonsterMyceliumRoot".to_string(),
        fields: vec![
            Field { name: "shard_id".to_string(), dtype: DataType::Int64 },
            Field { name: "prime".to_string(), dtype: DataType::Int64 },
            Field { name: "layer".to_string(), dtype: DataType::Int64 },
            Field { name: "harmonic_freq".to_string(), dtype: DataType::Float32 },
            Field { name: "lattice_coords".to_string(), dtype: DataType::Lattice71 },
            Field { name: "sample_data".to_string(), dtype: DataType::Float32 },
        ],
    }
}

/// Monster primes
const MONSTER_PRIMES: [u32; 71] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
    331, 337, 347, 349, 353,
];

/// Harmonic frequency for prime
fn harmonic_frequency(prime_idx: usize) -> f32 {
    440.0 * 2.0_f32.powf(prime_idx as f32 / 12.0)  // Chromatic scale
}

/// Extract first layer from shard
pub fn extract_first_layer(shard_path: PathBuf, shard_id: u8) -> Result<Vec<Spore>, Box<dyn std::error::Error>> {
    // Read parquet
    let df = ParquetReader::new(std::fs::File::open(shard_path)?)
        .finish()?;
    
    // Take first 10k rows (first layer)
    let first_layer = df.slice(0, 10_000);
    
    // Create spores (1000 samples per spore, 10 spores per shard)
    let mut spores = Vec::new();
    
    for i in 0..10 {
        let start = i * 1000;
        let end = start + 1000;
        let sample = first_layer.slice(start as i64, 1000);
        
        // Extract data
        let sample_data: Vec<f32> = sample.column("value")
            .unwrap()
            .f32()
            .unwrap()
            .into_iter()
            .map(|v| v.unwrap_or(0.0))
            .collect();
        
        // Compute lattice coordinates
        let lattice_coords = compute_lattice(&sample_data);
        
        spores.push(Spore {
            shard_id,
            prime: MONSTER_PRIMES[shard_id as usize],
            layer: 0,
            harmonic_freq: harmonic_frequency(shard_id as usize),
            sample_data,
            lattice_coords,
        });
    }
    
    Ok(spores)
}

fn compute_lattice(data: &[f32]) -> [i64; 71] {
    let mut coords = [0i64; 71];
    let sum: f32 = data.iter().sum();
    
    for i in 0..71 {
        coords[i] = (sum as i64) % (i as i64 + 1);
    }
    
    coords
}

/// Create spore jar from all shards
pub fn create_mycelium(shard_dir: PathBuf) -> Result<Vec<SporeJar>, Box<dyn std::error::Error>> {
    println!("🍄 Creating Mycelium Spore Jars");
    println!("="*70);
    println!();
    
    let mut jars = Vec::new();
    
    // Process each shard (71 total)
    for shard_id in 0..71 {
        let shard_path = shard_dir.join(format!("shard_{:02}.parquet", shard_id));
        
        if !shard_path.exists() {
            continue;
        }
        
        println!("  Processing shard {} (prime {})...", shard_id, MONSTER_PRIMES[shard_id]);
        
        let spores = extract_first_layer(shard_path, shard_id as u8)?;
        
        let jar = SporeJar {
            harmonic_id: shard_id,
            base_freq: harmonic_frequency(shard_id),
            spores,
            schema: root_schema(),
        };
        
        jars.push(jar);
    }
    
    println!();
    println!("✓ Created {} spore jars", jars.len());
    println!("✓ Total spores: {}", jars.iter().map(|j| j.spores.len()).sum::<usize>());
    
    Ok(jars)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_harmonic_frequencies() {
        let freq0 = harmonic_frequency(0);
        let freq1 = harmonic_frequency(1);
        
        assert!(freq1 > freq0);
        assert_eq!(freq0, 440.0);
    }
}
