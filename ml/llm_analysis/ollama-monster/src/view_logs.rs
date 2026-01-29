use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::collections::HashMap;

#[derive(Debug, Serialize)]
struct TraceLog {
    timestamp: String,
    prompt: String,
    response: String,
    prime_resonances: Vec<(u32, f64)>,
    top_registers: Vec<(String, u32, f64)>,
}

const PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

fn main() -> Result<()> {
    println!("📋 Full Trace Logs");
    println!("==================\n");
    
    let mut logs = Vec::new();
    
    for entry in fs::read_dir("perf_traces")? {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("prompt_") {
                let ts = name.strip_prefix("prompt_").unwrap().strip_suffix(".txt").unwrap();
                
                let prompt = fs::read_to_string(format!("perf_traces/prompt_{}.txt", ts))?;
                let response = fs::read_to_string(format!("perf_traces/response_{}.json", ts))?;
                let script = fs::read_to_string(format!("perf_traces/perf_regs_{}.script", ts))?;
                
                let response_text: serde_json::Value = serde_json::from_str(&response)?;
                let response_str = response_text["response"].as_str().unwrap_or("").to_string();
                
                let (primes, top_regs) = analyze(&script)?;
                
                logs.push(TraceLog {
                    timestamp: ts.to_string(),
                    prompt,
                    response: response_str,
                    prime_resonances: primes,
                    top_registers: top_regs,
                });
            }
        }
    }
    
    logs.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    
    for log in &logs {
        println!("=== {} ===", log.timestamp);
        println!("Prompt: {}", &log.prompt[..60.min(log.prompt.len())]);
        println!("Response: {}", &log.response[..80.min(log.response.len())]);
        println!("Primes: {:?}", &log.prime_resonances[..5]);
        println!("Top regs: {:?}", &log.top_registers[..3]);
        println!();
    }
    
    fs::write("FULL_TRACE_LOG.json", serde_json::to_string_pretty(&logs)?)?;
    println!("✓ Saved: FULL_TRACE_LOG.json ({} traces)", logs.len());
    
    Ok(())
}

fn analyze(script: &str) -> Result<(Vec<(u32, f64)>, Vec<(String, u32, f64)>)> {
    let mut register_values: HashMap<String, Vec<u64>> = HashMap::new();
    
    for line in script.lines() {
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
    
    let mut prime_counts: HashMap<u32, usize> = HashMap::new();
    let mut total = 0;
    
    for values in register_values.values() {
        for &value in values {
            total += 1;
            for &prime in &PRIMES {
                if value % prime as u64 == 0 {
                    *prime_counts.entry(prime).or_insert(0) += 1;
                }
            }
        }
    }
    
    let mut primes: Vec<_> = PRIMES.iter()
        .map(|&p| (p, *prime_counts.get(&p).unwrap_or(&0) as f64 / total as f64))
        .collect();
    primes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    let mut top_regs = Vec::new();
    for (reg, values) in &register_values {
        for &prime in &PRIMES {
            let count = values.iter().filter(|&&v| v % prime as u64 == 0).count();
            let pct = count as f64 / values.len() as f64;
            top_regs.push((reg.clone(), prime, pct));
        }
    }
    top_regs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    top_regs.truncate(10);
    
    Ok((primes, top_regs))
}
