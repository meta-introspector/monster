// Extract all constants from all Rust files and build value lattice

use std::fs;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use regex::Regex;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Constant {
    name: String,
    value: String,
    file: String,
    line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ValueLattice {
    constants: Vec<Constant>,
    value_map: HashMap<String, Vec<String>>,
    frequency: HashMap<String, usize>,
}

fn extract_constants_from_file(path: &str) -> Vec<Constant> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    
    let const_re = Regex::new(r"const\s+([A-Z_][A-Z0-9_]*)\s*:\s*\w+\s*=\s*([^;]+);").unwrap();
    let mut constants = Vec::new();
    
    for (line_num, line) in content.lines().enumerate() {
        if let Some(caps) = const_re.captures(line) {
            constants.push(Constant {
                name: caps[1].to_string(),
                value: caps[2].trim().to_string(),
                file: path.to_string(),
                line: line_num + 1,
            });
        }
    }
    
    constants
}

fn build_value_lattice(constants: Vec<Constant>) -> ValueLattice {
    let mut value_map: HashMap<String, Vec<String>> = HashMap::new();
    let mut frequency: HashMap<String, usize> = HashMap::new();
    
    for constant in &constants {
        value_map.entry(constant.value.clone())
            .or_default()
            .push(constant.name.clone());
        
        *frequency.entry(constant.value.clone()).or_default() += 1;
    }
    
    ValueLattice {
        constants,
        value_map,
        frequency,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 CONSTANT EXTRACTOR & VALUE LATTICE");
    println!("{}", "=".repeat(70));
    println!();
    
    let mut all_constants = Vec::new();
    
    // Extract from src/
    println!("📂 Scanning src/...");
    for entry in fs::read_dir("src")? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            let path_str = path.display().to_string();
            let constants = extract_constants_from_file(&path_str);
            
            if !constants.is_empty() {
                println!("  {}: {} constants", path_str, constants.len());
                all_constants.extend(constants);
            }
        }
    }
    
    // Extract from src/bin/
    for entry in fs::read_dir("src/bin")? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            let path_str = path.display().to_string();
            let constants = extract_constants_from_file(&path_str);
            
            if !constants.is_empty() {
                println!("  {}: {} constants", path_str, constants.len());
                all_constants.extend(constants);
            }
        }
    }
    
    println!();
    println!("📊 Building value lattice...");
    
    let lattice = build_value_lattice(all_constants);
    
    println!("  Total constants: {}", lattice.constants.len());
    println!("  Unique values: {}", lattice.value_map.len());
    
    println!();
    println!("🔢 Most Common Values:");
    println!("{}", "-".repeat(70));
    
    let mut freq_vec: Vec<_> = lattice.frequency.iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(a.1));
    
    for (value, count) in freq_vec.iter().take(10) {
        let names = &lattice.value_map[*value];
        println!("  {} ({}x): {:?}", 
            &value[..value.len().min(40)], count, &names[..names.len().min(3)]);
    }
    
    println!();
    println!("🎯 Key Constants:");
    println!("{}", "-".repeat(70));
    
    // Find important constants
    let key_patterns = ["24", "71", "BOSONIC", "MONSTER", "SHARD"];
    
    for pattern in &key_patterns {
        let matching: Vec<_> = lattice.constants.iter()
            .filter(|c| c.name.contains(pattern) || c.value.contains(pattern))
            .collect();
        
        if !matching.is_empty() {
            println!("\n  Pattern '{}': {} matches", pattern, matching.len());
            for c in matching.iter().take(5) {
                println!("    {} = {} ({})", c.name, 
                    &c.value[..c.value.len().min(30)], 
                    c.file.split('/').last().unwrap_or(&c.file));
            }
        }
    }
    
    println!();
    println!("💾 Saving lattice...");
    
    fs::create_dir_all("analysis")?;
    
    let json = serde_json::to_string_pretty(&lattice)?;
    fs::write("analysis/value_lattice.json", json)?;
    
    // Generate report
    let mut report = String::new();
    report.push_str("# Value Lattice Report\n\n");
    report.push_str(&format!("Total constants: {}\n", lattice.constants.len()));
    report.push_str(&format!("Unique values: {}\n\n", lattice.value_map.len()));
    
    report.push_str("## Most Common Values\n\n");
    for (value, count) in freq_vec.iter().take(20) {
        let names = &lattice.value_map[*value];
        report.push_str(&format!("- `{}` ({}x): {:?}\n", value, count, names));
    }
    
    report.push_str("\n## All Constants\n\n");
    for c in &lattice.constants {
        report.push_str(&format!("- `{}` = `{}` in `{}`\n", 
            c.name, c.value, c.file));
    }
    
    fs::write("analysis/VALUE_LATTICE_REPORT.md", report)?;
    
    println!("  ✅ analysis/value_lattice.json");
    println!("  ✅ analysis/VALUE_LATTICE_REPORT.md");
    
    println!();
    println!("✅ Value lattice refreshed!");
    
    Ok(())
}
