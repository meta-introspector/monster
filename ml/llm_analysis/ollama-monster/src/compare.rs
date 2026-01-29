use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;
use std::process::Command;
use std::fs;

#[derive(Debug, Serialize)]
struct Comparison {
    prompts: Vec<PromptResult>,
}

#[derive(Debug, Serialize)]
struct PromptResult {
    prompt: String,
    register_primes: HashMap<String, Vec<(u32, f64)>>,
}

const PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

#[tokio::main]
async fn main() -> Result<()> {
    let prompts = vec![
        "🌙",
        "red",
        "mathematician Conway",
    ];
    
    let mut results = Vec::new();
    
    for prompt in prompts {
        println!("Testing: {}", prompt);
        
        // Trace
        Command::new("./trace_regs.sh")
            .arg(prompt)
            .output()?;
        
        // Analyze latest
        let script = find_latest_script()?;
        let primes = analyze_registers(&script)?;
        
        results.push(PromptResult {
            prompt: prompt.to_string(),
            register_primes: primes,
        });
    }
    
    // Compare
    println!("\n📊 Comparison:");
    for result in &results {
        println!("\nPrompt: {}", result.prompt);
        for (reg, primes) in &result.register_primes {
            println!("  {}: {:?}", reg, &primes[..3]);
        }
    }
    
    fs::write("COMPARISON.json", serde_json::to_string_pretty(&Comparison { prompts: results })?)?;
    
    Ok(())
}

fn find_latest_script() -> Result<String> {
    let entries = fs::read_dir("perf_traces")?;
    let mut latest = None;
    let mut latest_time = std::time::SystemTime::UNIX_EPOCH;
    
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("script") {
            if let Ok(metadata) = entry.metadata() {
                if let Ok(modified) = metadata.modified() {
                    if modified > latest_time {
                        latest_time = modified;
                        latest = Some(path);
                    }
                }
            }
        }
    }
    
    Ok(fs::read_to_string(latest.unwrap())?)
}

fn analyze_registers(content: &str) -> Result<HashMap<String, Vec<(u32, f64)>>> {
    let mut register_values: HashMap<String, Vec<u64>> = HashMap::new();
    
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
    
    let mut result = HashMap::new();
    
    for (reg, values) in register_values {
        let mut prime_counts: Vec<(u32, f64)> = PRIMES.iter()
            .map(|&p| {
                let count = values.iter().filter(|&&v| v % p as u64 == 0).count();
                (p, count as f64 / values.len() as f64)
            })
            .collect();
        
        prime_counts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result.insert(reg, prime_counts);
    }
    
    Ok(result)
}
