use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;
use std::process::Command;
use std::fs;

#[derive(Debug, Serialize)]
struct FeedbackLoop {
    seed: String,
    iterations: Vec<Iteration>,
}

#[derive(Debug, Serialize)]
struct Iteration {
    iteration: usize,
    prompt: String,
    response: String,
    top_register_primes: Vec<(String, u32, f64)>,
}

const PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

#[tokio::main]
async fn main() -> Result<()> {
    let seeds = vec!["🌙", "red", "mathematician Conway"];
    let mut all_loops = Vec::new();
    
    for seed in seeds {
        println!("=== Seed: {} ===", seed);
        let loop_result = run_feedback_loop(seed).await?;
        all_loops.push(loop_result);
    }
    
    fs::write("FEEDBACK_LOOPS.json", serde_json::to_string_pretty(&all_loops)?)?;
    println!("\n✓ Saved: FEEDBACK_LOOPS.json");
    
    Ok(())
}

async fn run_feedback_loop(seed: &str) -> Result<FeedbackLoop> {
    let mut iterations = Vec::new();
    
    for i in 0..3 {
        let prompt = if i == 0 {
            seed.to_string()
        } else {
            let prev: &Iteration = &iterations[i - 1];
            format!(
                "Original: {}\n\nYour registers showed:\n{}\n\nWhat happens now?",
                seed,
                prev.top_register_primes.iter()
                    .take(3)
                    .map(|(r, p, s)| format!("  {}: {:.1}% prime {}", r, s * 100.0, p))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };
        
        println!("  Iteration {}: {}", i, &prompt[..40.min(prompt.len())]);
        
        // Query
        let response = query_model(&prompt).await?;
        
        // Trace
        Command::new("./trace_regs.sh")
            .arg(&prompt)
            .output()?;
        
        // Analyze
        let script = find_latest_script()?;
        let primes = analyze_registers(&script)?;
        
        // Get top 3
        let mut top: Vec<_> = primes.iter()
            .flat_map(|(reg, ps)| ps.iter().map(move |(p, s)| (reg.clone(), *p, *s)))
            .collect();
        top.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        top.truncate(5);
        
        println!("    Top: {:?}", &top[..3]);
        
        iterations.push(Iteration {
            iteration: i,
            prompt,
            response,
            top_register_primes: top,
        });
    }
    
    Ok(FeedbackLoop {
        seed: seed.to_string(),
        iterations,
    })
}

async fn query_model(prompt: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let response = client
        .post("http://localhost:11434/api/generate")
        .json(&serde_json::json!({
            "model": "qwen2.5:3b",
            "prompt": prompt,
            "stream": false
        }))
        .send()
        .await?;
    
    let result: serde_json::Value = response.json().await?;
    Ok(result["response"].as_str().unwrap_or("").to_string())
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
