use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Serialize)]
struct RegisterHistogram {
    register: String,
    bins: Vec<HistogramBin>,
    prime_patterns: HashMap<u32, Vec<u64>>,
}

#[derive(Debug, Serialize)]
struct HistogramBin {
    range_start: u64,
    range_end: u64,
    count: usize,
    values: Vec<u64>,
}

fn main() -> Result<()> {
    println!("📊 Register Value Histogram Analysis");
    println!("====================================\n");
    
    let script_path = "perf_traces/perf_regs_1769566105.script";
    let content = fs::read_to_string(script_path)?;
    
    let mut register_values: HashMap<String, Vec<u64>> = HashMap::new();
    
    // Parse register values
    for line in content.lines() {
        if line.contains("AX:") || line.contains("R8:") {
            for part in line.split_whitespace() {
                if let Some((reg, val)) = part.split_once(':') {
                    if let Some(hex) = val.strip_prefix("0x") {
                        if let Ok(value) = u64::from_str_radix(hex, 16) {
                            register_values.entry(reg.to_string())
                                .or_insert_with(Vec::new)
                                .push(value);
                        }
                    }
                }
            }
        }
    }
    
    println!("Found {} registers\n", register_values.len());
    
    let mut histograms = Vec::new();
    
    for (reg, values) in &register_values {
        if values.is_empty() {
            continue;
        }
        
        println!("Register {}: {} samples", reg, values.len());
        
        // Create histogram bins
        let min = *values.iter().min().unwrap();
        let max = *values.iter().max().unwrap();
        
        println!("  Range: 0x{:x} - 0x{:x}", min, max);
        
        // Group by prime divisibility
        let primes = [2u32, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
        let mut prime_patterns: HashMap<u32, Vec<u64>> = HashMap::new();
        
        for &value in values {
            for &prime in &primes {
                if value % prime as u64 == 0 {
                    prime_patterns.entry(prime)
                        .or_insert_with(Vec::new)
                        .push(value);
                }
            }
        }
        
        println!("  Prime divisibility:");
        for &prime in &primes {
            if let Some(vals) = prime_patterns.get(&prime) {
                println!("    Prime {}: {} values ({:.1}%)", 
                         prime, vals.len(), 
                         100.0 * vals.len() as f64 / values.len() as f64);
            }
        }
        
        // Show actual value distribution
        let mut sorted = values.clone();
        sorted.sort();
        
        println!("  Value distribution:");
        println!("    Min: 0x{:x} ({})", sorted[0], sorted[0]);
        println!("    P25: 0x{:x}", sorted[sorted.len() / 4]);
        println!("    P50: 0x{:x}", sorted[sorted.len() / 2]);
        println!("    P75: 0x{:x}", sorted[3 * sorted.len() / 4]);
        println!("    Max: 0x{:x} ({})", sorted[sorted.len() - 1], sorted[sorted.len() - 1]);
        
        // Show top 10 most common values
        let mut value_counts: HashMap<u64, usize> = HashMap::new();
        for &v in values {
            *value_counts.entry(v).or_insert(0) += 1;
        }
        
        let mut counts: Vec<_> = value_counts.iter().collect();
        counts.sort_by(|a, b| b.1.cmp(a.1));
        
        println!("  Top 10 values:");
        for (val, count) in counts.iter().take(10) {
            println!("    0x{:x}: {} times", val, count);
        }
        
        println!();
        
        histograms.push(RegisterHistogram {
            register: reg.clone(),
            bins: vec![],
            prime_patterns,
        });
    }
    
    fs::write(
        "REGISTER_HISTOGRAMS.json",
        serde_json::to_string_pretty(&histograms)?
    )?;
    
    println!("✓ Saved: REGISTER_HISTOGRAMS.json");
    
    Ok(())
}
