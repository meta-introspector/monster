use anyhow::Result;
use std::fs;
use std::path::PathBuf;
use serde::Serialize;

const MONSTER_PRIMES: [u32; 15] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
const MONSTER_DIGITS: [&str; 10] = ["8080", "1742", "479", "451", "2875", "8864", "5990", "496", "1710", "7570"];

#[derive(Debug, Serialize)]
struct ImagePattern {
    image: String,
    byte_patterns: Vec<BytePattern>,
    prime_resonances: Vec<u32>,
    symmetry_score: f64,
}

#[derive(Debug, Serialize)]
struct BytePattern {
    offset: usize,
    bytes: Vec<u8>,
    matches_monster: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎪 Monster Pattern Search in Images");
    println!("===================================\n");
    
    let invokeai_path = PathBuf::from("/mnt/data1/invokeai/outputs/images");
    
    let images: Vec<_> = fs::read_dir(&invokeai_path)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|s| s == "png").unwrap_or(false))
        .map(|e| e.path())
        .collect();
    
    println!("Found {} images to analyze\n", images.len());
    
    let mut all_patterns = Vec::new();
    
    for (i, img_path) in images.iter().enumerate() {
        println!("--- Image {} ---", i);
        println!("Path: {:?}", img_path.file_name().unwrap());
        
        let data = fs::read(&img_path)?;
        println!("Size: {} bytes", data.len());
        
        // Search for Monster patterns
        let patterns = find_monster_patterns(&data);
        let primes = find_prime_resonances(&data);
        let symmetry = calculate_symmetry(&data);
        
        println!("  Patterns found: {}", patterns.len());
        println!("  Prime resonances: {:?}", primes);
        println!("  Symmetry score: {:.3}", symmetry);
        
        all_patterns.push(ImagePattern {
            image: img_path.file_name().unwrap().to_string_lossy().to_string(),
            byte_patterns: patterns,
            prime_resonances: primes,
            symmetry_score: symmetry,
        });
        
        println!();
    }
    
    // Save results
    fs::write("IMAGE_PATTERNS.json", serde_json::to_string_pretty(&all_patterns)?)?;
    
    println!("✓ Search complete!");
    println!("\n📊 Summary:");
    println!("  Images analyzed: {}", all_patterns.len());
    println!("  Total patterns: {}", all_patterns.iter().map(|p| p.byte_patterns.len()).sum::<usize>());
    println!("  Results: IMAGE_PATTERNS.json");
    
    Ok(())
}

fn find_monster_patterns(data: &[u8]) -> Vec<BytePattern> {
    let mut patterns = Vec::new();
    
    // Search for 4-byte patterns (like "8080")
    for i in 0..data.len().saturating_sub(4) {
        let bytes = &data[i..i+4];
        
        // Check if matches Monster digits
        let s = format!("{:02x}{:02x}{:02x}{:02x}", bytes[0], bytes[1], bytes[2], bytes[3]);
        let matches = MONSTER_DIGITS.iter().any(|d| s.contains(d));
        
        if matches || is_symmetric(bytes) {
            patterns.push(BytePattern {
                offset: i,
                bytes: bytes.to_vec(),
                matches_monster: matches,
            });
        }
    }
    
    patterns
}

fn find_prime_resonances(data: &[u8]) -> Vec<u32> {
    let mut resonances = Vec::new();
    
    for &prime in &MONSTER_PRIMES {
        let count = data.iter().filter(|&&b| b as u32 % prime == 0).count();
        let ratio = count as f64 / data.len() as f64;
        
        if ratio > 0.1 {
            resonances.push(prime);
        }
    }
    
    resonances
}

fn calculate_symmetry(data: &[u8]) -> f64 {
    let mut score = 0.0;
    
    // Check for palindromic sections
    for chunk_size in [4, 8, 16] {
        for chunk in data.chunks(chunk_size) {
            if chunk == chunk.iter().rev().cloned().collect::<Vec<_>>() {
                score += 1.0;
            }
        }
    }
    
    score / (data.len() / 4) as f64
}

fn is_symmetric(bytes: &[u8]) -> bool {
    bytes[0] == bytes[3] && bytes[1] == bytes[2]
}
