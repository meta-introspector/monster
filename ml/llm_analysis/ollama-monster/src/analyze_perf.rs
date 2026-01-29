use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;

const MONSTER_PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

#[derive(Debug, Serialize)]
struct RegisterAnalysis {
    total_samples: usize,
    register_stats: HashMap<String, RegisterStats>,
    prime_resonances: HashMap<u32, f64>,
    hot_addresses: Vec<HotAddress>,
}

#[derive(Debug, Serialize)]
struct RegisterStats {
    register: String,
    mean: f64,
    max: u64,
    min: u64,
    prime_modulo_distribution: HashMap<u32, usize>,
}

#[derive(Debug, Serialize)]
struct HotAddress {
    address: String,
    count: usize,
    symbol: String,
}

fn main() -> Result<()> {
    println!("🔬 Analyzing perf register traces for Monster patterns");
    println!("====================================================\n");
    
    let perf_dir = "perf_traces";
    let entries = fs::read_dir(perf_dir)?;
    
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("script") {
            println!("Analyzing: {}", path.display());
            
            let content = fs::read_to_string(&path)?;
            let analysis = analyze_perf_script(&content)?;
            
            println!("  Total samples: {}", analysis.total_samples);
            println!("  Registers analyzed: {}", analysis.register_stats.len());
            
            // Show prime resonances
            println!("\n  Prime resonances:");
            let mut resonances: Vec<_> = analysis.prime_resonances.iter().collect();
            resonances.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            
            for (prime, strength) in resonances.iter().take(5) {
                println!("    Prime {}: {:.3}", prime, strength);
            }
            
            // Show hottest addresses
            println!("\n  Hottest addresses:");
            for hot in analysis.hot_addresses.iter().take(5) {
                println!("    {}: {} samples", hot.address, hot.count);
            }
            
            let output_file = path.with_extension("analysis.json");
            fs::write(&output_file, serde_json::to_string_pretty(&analysis)?)?;
            println!("\n  ✓ Saved: {}\n", output_file.display());
        }
    }
    
    Ok(())
}

fn analyze_perf_script(content: &str) -> Result<RegisterAnalysis> {
    let mut register_values: HashMap<String, Vec<u64>> = HashMap::new();
    let mut address_counts: HashMap<String, usize> = HashMap::new();
    let mut total_samples = 0;
    
    for line in content.lines() {
        // Parse register values: AX:0x1234 BX:0x5678 ...
        if line.contains("AX:") || line.contains("R8:") {
            total_samples += 1;
            
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
        
        // Parse addresses
        if line.starts_with("    ") && line.contains("0x") {
            if let Some(addr) = line.split_whitespace().next() {
                *address_counts.entry(addr.to_string()).or_insert(0) += 1;
            }
        }
    }
    
    // Calculate register statistics
    let mut register_stats = HashMap::new();
    
    for (reg, values) in &register_values {
        if values.is_empty() {
            continue;
        }
        
        let sum: u64 = values.iter().sum();
        let mean = sum as f64 / values.len() as f64;
        let max = *values.iter().max().unwrap();
        let min = *values.iter().min().unwrap();
        
        // Calculate prime modulo distribution
        let mut prime_modulo_distribution = HashMap::new();
        for &value in values {
            for &prime in &MONSTER_PRIMES {
                let modulo = (value % prime as u64) as u32;
                if modulo == 0 {
                    *prime_modulo_distribution.entry(prime).or_insert(0) += 1;
                }
            }
        }
        
        register_stats.insert(reg.clone(), RegisterStats {
            register: reg.clone(),
            mean,
            max,
            min,
            prime_modulo_distribution,
        });
    }
    
    // Calculate overall prime resonances
    let mut prime_resonances = HashMap::new();
    for &prime in &MONSTER_PRIMES {
        let mut total_hits = 0;
        let mut total_values = 0;
        
        for stats in register_stats.values() {
            if let Some(&hits) = stats.prime_modulo_distribution.get(&prime) {
                total_hits += hits;
            }
            total_values += register_values[&stats.register].len();
        }
        
        let resonance = if total_values > 0 {
            total_hits as f64 / total_values as f64
        } else {
            0.0
        };
        
        prime_resonances.insert(prime, resonance);
    }
    
    // Get hot addresses
    let mut hot_addresses: Vec<_> = address_counts.into_iter()
        .map(|(addr, count)| HotAddress {
            address: addr,
            count,
            symbol: "[unknown]".to_string(),
        })
        .collect();
    hot_addresses.sort_by(|a, b| b.count.cmp(&a.count));
    hot_addresses.truncate(20);
    
    Ok(RegisterAnalysis {
        total_samples,
        register_stats,
        prime_resonances,
        hot_addresses,
    })
}
