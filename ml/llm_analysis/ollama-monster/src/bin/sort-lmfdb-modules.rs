//! Sort LMFDB modules into 71 Monster shards by prime resonance
//! Analyzes Python source code and mathematical objects

use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const MONSTER_PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

#[derive(Debug)]
struct LMFDBModule {
    name: String,
    path: PathBuf,
    prime_resonances: Vec<(u32, f64)>,
    primary_shard: u32,
}

fn main() -> Result<()> {
    println!("🔷 Sorting LMFDB Modules by Prime Resonance");
    println!("============================================\n");
    
    let lmfdb_root = Path::new("/mnt/data1/nix/source/github/meta-introspector/lmfdb/lmfdb");
    let output_dir = Path::new("/home/mdupont/experiments/monster/monster-shards");
    
    let modules = scan_lmfdb_modules(lmfdb_root)?;
    
    println!("Found {} LMFDB modules\n", modules.len());
    
    let mut shard_counts: HashMap<u32, usize> = HashMap::new();
    
    for module in &modules {
        println!("Module: {}", module.name);
        println!("  Primary shard: {}", module.primary_shard);
        println!("  Resonances: {:?}", module.prime_resonances);
        
        *shard_counts.entry(module.primary_shard).or_insert(0) += 1;
        
        // Copy module to shard
        copy_to_shard(module, output_dir)?;
    }
    
    println!("\n============================================");
    println!("Summary:\n");
    
    let mut shards: Vec<_> = shard_counts.iter().collect();
    shards.sort_by_key(|(k, _)| *k);
    
    for (shard, count) in shards {
        let marker = if MONSTER_PRIMES.contains(shard) { "★" } else { " " };
        println!("  Shard {:2} {}: {:2} modules", shard, marker, count);
    }
    
    println!("\n✅ LMFDB sorted into {} shards!", shard_counts.len());
    
    Ok(())
}

fn scan_lmfdb_modules(root: &Path) -> Result<Vec<LMFDBModule>> {
    let mut modules = Vec::new();
    
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            
            // Skip special directories
            if name.starts_with('.') || name == "__pycache__" {
                continue;
            }
            
            let resonances = analyze_module(&path)?;
            let primary_shard = determine_primary_shard(&resonances);
            
            modules.push(LMFDBModule {
                name,
                path,
                prime_resonances: resonances,
                primary_shard,
            });
        }
    }
    
    Ok(modules)
}

fn analyze_module(path: &Path) -> Result<Vec<(u32, f64)>> {
    let mut prime_counts: HashMap<u32, usize> = HashMap::new();
    let mut total_numbers = 0;
    
    // Walk through all Python files
    for entry in WalkDir::new(path).max_depth(3) {
        let entry = entry?;
        if entry.path().extension().and_then(|s| s.to_str()) == Some("py") {
            if let Ok(content) = fs::read_to_string(entry.path()) {
                let numbers = extract_numbers_from_code(&content);
                total_numbers += numbers.len();
                
                for num in numbers {
                    for &prime in &MONSTER_PRIMES {
                        if num % prime == 0 {
                            *prime_counts.entry(prime).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
    }
    
    if total_numbers == 0 {
        return Ok(vec![]);
    }
    
    let mut resonances: Vec<_> = prime_counts.iter()
        .map(|(&prime, &count)| (prime, count as f64 / total_numbers as f64))
        .collect();
    
    resonances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    Ok(resonances)
}

fn extract_numbers_from_code(code: &str) -> Vec<u32> {
    let mut numbers = Vec::new();
    
    // Simple regex-like extraction of numbers
    let mut current_num = String::new();
    
    for ch in code.chars() {
        if ch.is_ascii_digit() {
            current_num.push(ch);
        } else if !current_num.is_empty() {
            if let Ok(num) = current_num.parse::<u32>() {
                if num > 0 && num < 1000000 {
                    numbers.push(num);
                }
            }
            current_num.clear();
        }
    }
    
    numbers
}

fn determine_primary_shard(resonances: &[(u32, f64)]) -> u32 {
    if resonances.is_empty() {
        return 1; // Default
    }
    
    resonances[0].0
}

fn copy_to_shard(module: &LMFDBModule, output_dir: &Path) -> Result<()> {
    let shard_dir = output_dir
        .join(format!("shard-{:02}", module.primary_shard))
        .join("data")
        .join("lmfdb")
        .join(&module.name);
    
    fs::create_dir_all(&shard_dir)?;
    
    // Create a metadata file
    let metadata = format!(
        "# LMFDB Module: {}\n\
         Primary Shard: {}\n\
         Prime Resonances: {:?}\n\
         Source: {:?}\n",
        module.name,
        module.primary_shard,
        module.prime_resonances,
        module.path
    );
    
    fs::write(shard_dir.join("README.md"), metadata)?;
    
    Ok(())
}
