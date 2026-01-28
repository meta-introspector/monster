use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use memmap2::Mmap;

/// Monster primes and their powers
const MONSTER_PRIMES: [(u32, u32); 15] = [
    (2, 46), (3, 20), (5, 9), (7, 6), (11, 2), (13, 3),
    (17, 1), (19, 1), (23, 1), (29, 1), (31, 1), (41, 1),
    (47, 1), (59, 1), (71, 1)
];

/// Monster leading digits
const MONSTER_DIGITS: [&str; 10] = [
    "8080", "1742", "479", "451", "2875", "8864", "5990", "496", "1710", "7570"
];

/// N-gram pattern with Monster symmetry
#[derive(Debug, Serialize)]
struct MonsterPattern {
    ngram: Vec<u8>,
    size: usize,
    frequency: usize,
    prime_signature: Vec<u32>,
    symmetry_score: f64,
    locations: Vec<usize>,
}

/// Memory model of Ollama weights
struct ModelMemory {
    data: Vec<u8>,
    size: usize,
    ngram_index: HashMap<Vec<u8>, Vec<usize>>,
}

impl ModelMemory {
    fn load(path: &str, max_size: usize) -> Result<Self> {
        println!("📂 Loading model from: {}", path);
        
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        
        let bytes_read = file.take(max_size as u64).read_to_end(&mut data)?;
        
        println!("   ✓ Loaded {} bytes ({:.2} GB)", bytes_read, bytes_read as f64 / 1e9);
        
        Ok(Self {
            data,
            size: bytes_read,
            ngram_index: HashMap::new(),
        })
    }
    
    /// Build n-gram index with depth p
    fn build_ngram_index(&mut self, n: usize, depth: usize) {
        println!("\n🔍 Building {}-gram index (depth: {})...", n, depth);
        
        let mut count = 0;
        for i in 0..self.data.len().saturating_sub(n) {
            if count >= depth { break; }
            
            let ngram = self.data[i..i+n].to_vec();
            self.ngram_index.entry(ngram)
                .or_insert_with(Vec::new)
                .push(i);
            
            count += 1;
            
            if count % 1_000_000 == 0 {
                println!("   Indexed {} n-grams...", count);
            }
        }
        
        println!("   ✓ Indexed {} unique {}-grams", self.ngram_index.len(), n);
    }
    
    /// Search for Monster symmetry patterns
    fn find_monster_patterns(&self, n: usize) -> Vec<MonsterPattern> {
        println!("\n🎪 Searching for Monster symmetry patterns...");
        
        let mut patterns = Vec::new();
        
        for (ngram, locations) in &self.ngram_index {
            if ngram.len() != n { continue; }
            
            // Calculate prime signature
            let prime_sig = self.extract_prime_signature(ngram);
            
            // Calculate symmetry score
            let symmetry = self.calculate_symmetry(ngram, &prime_sig);
            
            // Check if matches Monster patterns
            if symmetry > 0.5 || self.matches_monster_digits(ngram) {
                patterns.push(MonsterPattern {
                    ngram: ngram.clone(),
                    size: n,
                    frequency: locations.len(),
                    prime_signature: prime_sig,
                    symmetry_score: symmetry,
                    locations: locations.clone(),
                });
            }
        }
        
        // Sort by symmetry score
        patterns.sort_by(|a, b| b.symmetry_score.partial_cmp(&a.symmetry_score).unwrap());
        
        println!("   ✓ Found {} Monster patterns", patterns.len());
        
        patterns
    }
    
    fn extract_prime_signature(&self, ngram: &[u8]) -> Vec<u32> {
        let mut primes = Vec::new();
        
        for &byte in ngram {
            for &(prime, _) in &MONSTER_PRIMES {
                if byte as u32 % prime == 0 {
                    if !primes.contains(&prime) {
                        primes.push(prime);
                    }
                }
            }
        }
        
        primes
    }
    
    fn calculate_symmetry(&self, ngram: &[u8], primes: &[u32]) -> f64 {
        // Check for Monster prime patterns
        let monster_prime_match = primes.iter()
            .filter(|p| MONSTER_PRIMES.iter().any(|(mp, _)| mp == *p))
            .count() as f64 / MONSTER_PRIMES.len() as f64;
        
        // Check for palindromic symmetry
        let palindrome_score = if ngram == ngram.iter().rev().cloned().collect::<Vec<_>>() {
            1.0
        } else {
            0.0
        };
        
        // Check for repeating patterns (like 8080)
        let repeat_score = if ngram.len() >= 4 {
            let half = ngram.len() / 2;
            if ngram[..half] == ngram[half..2*half] {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        (monster_prime_match + palindrome_score + repeat_score) / 3.0
    }
    
    fn matches_monster_digits(&self, ngram: &[u8]) -> bool {
        let s = String::from_utf8_lossy(ngram);
        MONSTER_DIGITS.iter().any(|d| s.contains(d))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎪 Monster Symmetry Search in Ollama Model");
    println!("==========================================\n");
    
    // Find qwen2.5:3b model file
    let model_path = find_ollama_model("qwen2.5:3b")?;
    
    println!("Model: qwen2.5:3b");
    println!("Available RAM: 30 GB\n");
    
    // Load model into memory (use 20GB max)
    let max_size = 20 * 1024 * 1024 * 1024; // 20 GB
    let mut memory = ModelMemory::load(&model_path, max_size)?;
    
    // Parameters: n-gram size, depth
    let configs = vec![
        (4, 10_000_000),  // 4-grams, 10M depth (like "8080")
        (8, 5_000_000),   // 8-grams, 5M depth
        (16, 2_000_000),  // 16-grams, 2M depth
    ];
    
    let mut all_patterns = Vec::new();
    
    for (n, depth) in configs {
        println!("\n=== N-gram size: {}, Depth: {} ===", n, depth);
        
        // Build index
        memory.build_ngram_index(n, depth);
        
        // Find Monster patterns
        let patterns = memory.find_monster_patterns(n);
        
        // Show top 10
        println!("\nTop 10 Monster patterns:");
        for (i, pattern) in patterns.iter().take(10).enumerate() {
            println!("  {}. Symmetry: {:.3}, Freq: {}, Primes: {:?}",
                     i+1, pattern.symmetry_score, pattern.frequency, pattern.prime_signature);
            println!("     Bytes: {:?}", &pattern.ngram[..pattern.ngram.len().min(16)]);
        }
        
        all_patterns.extend(patterns);
    }
    
    // Save results
    let report = serde_json::to_string_pretty(&all_patterns)?;
    std::fs::write("MONSTER_PATTERNS.json", report)?;
    
    println!("\n✓ Search complete!");
    println!("\n📊 Summary:");
    println!("  Total patterns found: {}", all_patterns.len());
    println!("  Model size analyzed: {:.2} GB", memory.size as f64 / 1e9);
    println!("  Results saved to: MONSTER_PATTERNS.json");
    
    Ok(())
}

fn find_ollama_model(name: &str) -> Result<String> {
    // Ollama stores models in ~/.ollama/models/blobs/
    let home = std::env::var("HOME")?;
    let ollama_dir = format!("{}/.ollama/models/blobs", home);
    
    println!("🔍 Searching for model in: {}", ollama_dir);
    
    // Find largest blob (the model weights)
    let mut largest = None;
    let mut largest_size = 0;
    
    for entry in std::fs::read_dir(&ollama_dir)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        
        if metadata.len() > largest_size {
            largest_size = metadata.len();
            largest = Some(entry.path());
        }
    }
    
    if let Some(path) = largest {
        println!("   ✓ Found model: {} ({:.2} GB)", 
                 path.display(), largest_size as f64 / 1e9);
        Ok(path.to_string_lossy().to_string())
    } else {
        anyhow::bail!("Model not found")
    }
}
