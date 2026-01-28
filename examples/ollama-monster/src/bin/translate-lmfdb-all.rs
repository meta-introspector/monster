//! Translate ALL LMFDB modules to Rust + Lean4
//! Distribute across 71 Monster shards by prime resonance

use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

const MONSTER_PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

fn main() -> Result<()> {
    println!("🔷 Translating ALL LMFDB to Rust + Lean4");
    println!("=========================================\n");
    
    let lmfdb_root = Path::new("/mnt/data1/nix/source/github/meta-introspector/lmfdb/lmfdb");
    let output_base = Path::new("/home/mdupont/experiments/monster/monster-shards");
    
    let mut stats = TranslationStats::new();
    
    // Get all LMFDB modules
    for entry in fs::read_dir(lmfdb_root)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_dir() {
            let module_name = path.file_name().unwrap().to_string_lossy().to_string();
            
            if module_name.starts_with('.') || module_name == "__pycache__" {
                continue;
            }
            
            println!("📦 Module: {}", module_name);
            
            let module_stats = translate_module(&path, &module_name, output_base)?;
            stats.merge(module_stats);
            
            println!();
        }
    }
    
    stats.print_summary();
    
    Ok(())
}

struct TranslationStats {
    rust_files: HashMap<u32, usize>,
    lean_files: HashMap<u32, usize>,
    total_lines: usize,
}

impl TranslationStats {
    fn new() -> Self {
        Self {
            rust_files: HashMap::new(),
            lean_files: HashMap::new(),
            total_lines: 0,
        }
    }
    
    fn merge(&mut self, other: TranslationStats) {
        for (shard, count) in other.rust_files {
            *self.rust_files.entry(shard).or_insert(0) += count;
        }
        for (shard, count) in other.lean_files {
            *self.lean_files.entry(shard).or_insert(0) += count;
        }
        self.total_lines += other.total_lines;
    }
    
    fn print_summary(&self) {
        println!("\n=========================================");
        println!("📊 TRANSLATION SUMMARY\n");
        
        println!("Rust files by shard:");
        let mut rust: Vec<_> = self.rust_files.iter().collect();
        rust.sort_by_key(|(k, _)| *k);
        for (shard, count) in rust {
            let marker = if MONSTER_PRIMES.contains(shard) { "★" } else { " " };
            println!("  Shard {:2} {}: {} files", shard, marker, count);
        }
        
        println!("\nLean4 files by shard:");
        let mut lean: Vec<_> = self.lean_files.iter().collect();
        lean.sort_by_key(|(k, _)| *k);
        for (shard, count) in lean {
            let marker = if MONSTER_PRIMES.contains(shard) { "★" } else { " " };
            println!("  Shard {:2} {}: {} files", shard, marker, count);
        }
        
        let total_rust: usize = self.rust_files.values().sum();
        let total_lean: usize = self.lean_files.values().sum();
        
        println!("\n✅ Translation complete!");
        println!("   Rust files: {}", total_rust);
        println!("   Lean4 files: {}", total_lean);
        println!("   Total lines: {}", self.total_lines);
        println!("   Shards used: {}", self.rust_files.len());
    }
}

fn translate_module(path: &Path, module_name: &str, output_base: &Path) -> Result<TranslationStats> {
    let mut stats = TranslationStats::new();
    
    for entry in WalkDir::new(path).max_depth(3) {
        let entry = entry?;
        let file_path = entry.path();
        
        if file_path.extension().and_then(|s| s.to_str()) == Some("py") {
            let filename = file_path.file_name().unwrap().to_string_lossy();
            
            if filename.starts_with("__") || filename.starts_with("test_") {
                continue;
            }
            
            let python_code = fs::read_to_string(file_path)?;
            let lines = python_code.lines().count();
            stats.total_lines += lines;
            
            // Calculate prime resonance
            let numbers = extract_numbers(&python_code);
            let shard = calculate_prime_resonance(&numbers);
            
            println!("  {} → Shard {}", filename, shard);
            
            // Translate to Rust
            let rust_code = python_to_rust(&python_code, module_name, &filename)?;
            write_rust_file(output_base, shard, module_name, &filename, &rust_code)?;
            *stats.rust_files.entry(shard).or_insert(0) += 1;
            
            // Translate to Lean4
            let lean_code = python_to_lean4(&python_code, module_name, &filename)?;
            write_lean_file(output_base, shard, module_name, &filename, &lean_code)?;
            *stats.lean_files.entry(shard).or_insert(0) += 1;
        }
    }
    
    Ok(stats)
}

fn extract_numbers(code: &str) -> Vec<u32> {
    let mut numbers = Vec::new();
    let mut current = String::new();
    
    for ch in code.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            if let Ok(num) = current.parse::<u32>() {
                if num > 0 && num < 1000000 {
                    numbers.push(num);
                }
            }
            current.clear();
        }
    }
    
    numbers
}

fn calculate_prime_resonance(numbers: &[u32]) -> u32 {
    if numbers.is_empty() {
        return 1;
    }
    
    let mut best_prime = 2;
    let mut best_score = 0.0;
    
    for &prime in &MONSTER_PRIMES {
        let score = numbers.iter()
            .filter(|&&n| n % prime == 0)
            .count() as f64 / numbers.len() as f64;
        
        if score > best_score {
            best_score = score;
            best_prime = prime;
        }
    }
    
    best_prime
}

fn python_to_rust(python_code: &str, module: &str, filename: &str) -> Result<String> {
    let mut rust_code = format!(
        "//! LMFDB {} - Translated from {}\n\
         //! Monster Shard - Prime resonance distribution\n\n\
         use serde::{{Serialize, Deserialize}};\n\
         use std::collections::HashMap;\n\n",
        module, filename
    );
    
    // Extract classes → structs
    for line in python_code.lines() {
        if line.trim().starts_with("class ") {
            if let Some(name) = extract_class_name(line) {
                rust_code.push_str(&format!("#[derive(Debug, Serialize, Deserialize)]\n"));
                rust_code.push_str(&format!("pub struct {} {{\n", name));
                rust_code.push_str("    // TODO: Translate fields\n");
                rust_code.push_str("}\n\n");
            }
        }
        
        if line.trim().starts_with("def ") {
            if let Some(name) = extract_function_name(line) {
                rust_code.push_str(&format!("pub fn {}() {{\n", name));
                rust_code.push_str("    // TODO: Translate function body\n");
                rust_code.push_str("    todo!()\n");
                rust_code.push_str("}\n\n");
            }
        }
    }
    
    Ok(rust_code)
}

fn python_to_lean4(python_code: &str, module: &str, filename: &str) -> Result<String> {
    let mut lean_code = format!(
        "-- LMFDB {} - Translated from {}\n\
         -- Monster Shard - Prime resonance distribution\n\n\
         import Mathlib.Data.Nat.Prime\n\
         import Mathlib.NumberTheory.NumberField.Basic\n\n\
         namespace LMFDB.{}\n\n",
        module, filename, module.replace("_", "")
    );
    
    // Extract classes → structures
    for line in python_code.lines() {
        if line.trim().starts_with("class ") {
            if let Some(name) = extract_class_name(line) {
                lean_code.push_str(&format!("structure {} where\n", name));
                lean_code.push_str("  -- TODO: Translate fields\n");
                lean_code.push_str("  sorry : Unit\n\n");
            }
        }
        
        if line.trim().starts_with("def ") {
            if let Some(name) = extract_function_name(line) {
                lean_code.push_str(&format!("def {} : Unit := sorry\n\n", name));
            }
        }
    }
    
    lean_code.push_str(&format!("end LMFDB.{}\n", module.replace("_", "")));
    
    Ok(lean_code)
}

fn extract_class_name(line: &str) -> Option<String> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 2 && parts[0] == "class" {
        let name = parts[1].trim_end_matches(':').trim_end_matches('(');
        Some(name.to_string())
    } else {
        None
    }
}

fn extract_function_name(line: &str) -> Option<String> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 2 && parts[0] == "def" {
        let name = parts[1].split('(').next()?;
        Some(name.to_string())
    } else {
        None
    }
}

fn write_rust_file(base: &Path, shard: u32, module: &str, filename: &str, code: &str) -> Result<()> {
    let dir = base.join(format!("shard-{:02}", shard)).join("rust").join(module);
    fs::create_dir_all(&dir)?;
    
    let stem = Path::new(filename).file_stem().unwrap().to_string_lossy();
    let file = dir.join(format!("{}.rs", stem));
    fs::write(file, code)?;
    
    Ok(())
}

fn write_lean_file(base: &Path, shard: u32, module: &str, filename: &str, code: &str) -> Result<()> {
    let dir = base.join(format!("shard-{:02}", shard)).join("lean4").join(module);
    fs::create_dir_all(&dir)?;
    
    let stem = Path::new(filename).file_stem().unwrap().to_string_lossy();
    let file = dir.join(format!("{}.lean", stem));
    fs::write(file, code)?;
    
    Ok(())
}
