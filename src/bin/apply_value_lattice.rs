// Apply zos-server value lattice to Monster project

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
struct ValueLatticeEntry {
    value: String,
    godel_number: u64,
    usage_count: u32,
    file_locations: Vec<String>,
}

fn extract_values_from_file(path: &PathBuf) -> Vec<String> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    
    let mut values = Vec::new();
    let chars: Vec<char> = content.chars().collect();
    let mut i = 0;
    
    while i < chars.len() {
        if chars[i].is_ascii_digit() || chars[i] == '-' {
            let start = i;
            if chars[i] == '-' { i += 1; }
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let number_str: String = chars[start..i].iter().collect();
            if number_str.parse::<i64>().is_ok() {
                values.push(number_str);
            }
        } else {
            i += 1;
        }
    }
    
    values
}

fn build_lattice() -> HashMap<String, ValueLatticeEntry> {
    let mut lattice = HashMap::new();
    let mut godel = 1u64;
    
    // Scan src/
    for entry in fs::read_dir("src").unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            let values = extract_values_from_file(&path);
            let file_str = path.display().to_string();
            
            for value in values {
                lattice.entry(value.clone())
                    .and_modify(|e: &mut ValueLatticeEntry| {
                        e.usage_count += 1;
                        if !e.file_locations.contains(&file_str) {
                            e.file_locations.push(file_str.clone());
                        }
                    })
                    .or_insert_with(|| {
                        let entry = ValueLatticeEntry {
                            value: value.clone(),
                            godel_number: godel,
                            usage_count: 1,
                            file_locations: vec![file_str.clone()],
                        };
                        godel += 1;
                        entry
                    });
            }
        }
    }
    
    // Scan src/bin/
    for entry in fs::read_dir("src/bin").unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            let values = extract_values_from_file(&path);
            let file_str = path.display().to_string();
            
            for value in values {
                lattice.entry(value.clone())
                    .and_modify(|e| {
                        e.usage_count += 1;
                        if !e.file_locations.contains(&file_str) {
                            e.file_locations.push(file_str.clone());
                        }
                    })
                    .or_insert_with(|| {
                        let entry = ValueLatticeEntry {
                            value: value.clone(),
                            godel_number: godel,
                            usage_count: 1,
                            file_locations: vec![file_str.clone()],
                        };
                        godel += 1;
                        entry
                    });
            }
        }
    }
    
    lattice
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 VALUE LATTICE (zos-server method)");
    println!("{}", "=".repeat(70));
    println!();
    
    println!("📂 Building lattice from Monster project...");
    let lattice = build_lattice();
    
    println!("  Total unique values: {}", lattice.len());
    println!();
    
    // Top values by usage
    let mut values: Vec<_> = lattice.values().collect();
    values.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));
    
    println!("🔢 Top 20 Values by Usage:");
    println!("{}", "-".repeat(70));
    for entry in values.iter().take(20) {
        println!("  {} (Gödel: {}, {}x in {} files)", 
            entry.value, entry.godel_number, entry.usage_count, entry.file_locations.len());
    }
    
    println!();
    println!("🎯 Monster Constants:");
    println!("{}", "-".repeat(70));
    
    let key_values = ["24", "71", "46", "20", "9", "6", "2", "3"];
    for val in &key_values {
        if let Some(entry) = lattice.get(*val) {
            println!("  {} (Gödel: {}, {}x): {:?}", 
                entry.value, entry.godel_number, entry.usage_count, 
                &entry.file_locations[..entry.file_locations.len().min(3)]);
        }
    }
    
    println!();
    println!("💾 Saving lattice...");
    
    fs::create_dir_all("analysis")?;
    
    let json = serde_json::to_string_pretty(&lattice)?;
    fs::write("analysis/value_lattice_full.json", json)?;
    
    println!("  ✅ analysis/value_lattice_full.json");
    println!();
    println!("✅ Value lattice applied!");
    
    Ok(())
}
