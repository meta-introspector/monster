//! Translate LMFDB Hilbert Modular Forms to Lean4
//! Distribute across 71 Monster shards by prime resonance

use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

const MONSTER_PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

const LEAN4_HEADER: &str = r#"-- Hilbert Modular Forms in Lean4
-- Monster Shard {shard}: Prime {prime} resonance
-- Translated from LMFDB Python code

import Mathlib.NumberTheory.ModularForms.Basic
import Mathlib.NumberTheory.NumberField.Basic
import Mathlib.RingTheory.DedekindDomain.Ideal

namespace HilbertModularForms

"#;

fn main() -> Result<()> {
    println!("🔷 Translating Hilbert Modular Forms to Lean4");
    println!("==============================================\n");
    
    let lmfdb_path = Path::new("/mnt/data1/nix/source/github/meta-introspector/lmfdb/lmfdb/hilbert_modular_forms");
    let output_base = Path::new("/home/mdupont/experiments/monster/monster-shards");
    
    let mut shard_counts: HashMap<u32, usize> = HashMap::new();
    
    for entry in WalkDir::new(lmfdb_path).max_depth(1) {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("py") {
            let filename = path.file_name().unwrap().to_string_lossy();
            
            if filename.starts_with("__") {
                continue;
            }
            
            println!("Processing {}...", filename);
            
            let python_code = fs::read_to_string(path)?;
            
            // Calculate prime resonance
            let numbers = extract_numbers(&python_code);
            let shard = calculate_prime_resonance(&numbers);
            
            println!("  → Shard {} (prime resonance)", shard);
            
            // Translate to Lean4
            let lean_code = python_to_lean4(&python_code, shard, &filename)?;
            
            // Write to shard
            let shard_dir = output_base
                .join(format!("shard-{:02}", shard))
                .join("lean4")
                .join("HilbertModularForms");
            
            fs::create_dir_all(&shard_dir)?;
            
            let stem = path.file_stem().unwrap().to_string_lossy();
            let lean_file = shard_dir.join(format!("{}.lean", stem));
            fs::write(&lean_file, lean_code)?;
            
            *shard_counts.entry(shard).or_insert(0) += 1;
            println!("  ✓ Written to {:?}\n", lean_file);
        }
    }
    
    println!("==============================================");
    println!("Summary:\n");
    
    let mut shards: Vec<_> = shard_counts.iter().collect();
    shards.sort_by_key(|(k, _)| *k);
    
    for (shard, count) in shards {
        let marker = if MONSTER_PRIMES.contains(shard) { "★" } else { " " };
        println!("  Shard {:2} {}: {} files", shard, marker, count);
    }
    
    let total: usize = shard_counts.values().sum();
    println!("\n✅ Hilbert Modular Forms translated to Lean4!");
    println!("   Total: {} files across {} shards", total, shard_counts.len());
    
    Ok(())
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

fn python_to_lean4(python_code: &str, shard: u32, filename: &str) -> Result<String> {
    let mut lean_code = LEAN4_HEADER
        .replace("{shard}", &shard.to_string())
        .replace("{prime}", &shard.to_string());
    
    lean_code.push_str(&format!("-- Translated from: {}\n\n", filename));
    
    // Extract class definitions
    for line in python_code.lines() {
        if line.trim().starts_with("class ") {
            if let Some(class_name) = extract_class_name(line) {
                lean_code.push_str(&format!("structure {} where\n", class_name));
                lean_code.push_str(&format!("  -- TODO: Translate fields from Python class {}\n", class_name));
                lean_code.push_str("  sorry : Unit\n\n");
            }
        }
        
        if line.trim().starts_with("def ") {
            if let Some(func_name) = extract_function_name(line) {
                lean_code.push_str(&format!("def {} : Unit := sorry\n", func_name));
                lean_code.push_str(&format!("  -- TODO: Translate function {}\n\n", func_name));
            }
        }
    }
    
    lean_code.push_str("end HilbertModularForms\n");
    
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
