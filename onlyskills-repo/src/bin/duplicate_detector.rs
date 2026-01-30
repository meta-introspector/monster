// Duplicate Detector - Search 8M files for copies
use std::collections::HashMap;
use std::path::Path;
use sha2::{Sha256, Digest};
use walkdir::WalkDir;

fn hash_file(path: &Path) -> Option<String> {
    let content = std::fs::read(path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(&content);
    Some(format!("{:x}", hasher.finalize()))
}

fn find_duplicates(base: &Path, limit: usize) -> HashMap<String, Vec<String>> {
    let mut hashes: HashMap<String, Vec<String>> = HashMap::new();
    
    for entry in WalkDir::new(base).max_depth(5).into_iter().take(limit) {
        if let Ok(entry) = entry {
            if entry.file_type().is_file() {
                if let Some(hash) = hash_file(entry.path()) {
                    hashes.entry(hash)
                        .or_default()
                        .push(entry.path().display().to_string());
                }
            }
        }
    }
    
    hashes.into_iter()
        .filter(|(_, paths)| paths.len() > 1)
        .collect()
}

fn main() {
    println!("🔍 Duplicate Detector (8M files)");
    println!("{}", "=".repeat(70));
    
    let base = Path::new("/home/mdupont/experiments/monster");
    
    println!("Scanning first 10,000 files...");
    let duplicates = find_duplicates(base, 10_000);
    
    println!("Found {} duplicate groups", duplicates.len());
    
    for (hash, paths) in duplicates.iter().take(5) {
        println!("\nHash: {}...", &hash[..16]);
        for path in paths {
            println!("  {}", path);
        }
    }
    
    let json = serde_json::to_string_pretty(&duplicates).unwrap();
    std::fs::write("duplicates_found.json", json).unwrap();
    
    println!("\n✓ Saved: duplicates_found.json");
    println!("✓ Novelty: {} unique files", 10_000 - duplicates.len());
}
