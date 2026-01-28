use anyhow::Result;
use serde::Serialize;
use std::fs;

#[derive(Debug, Serialize)]
struct EigenvectorSearch {
    iterations: Vec<EigenIteration>,
    converged: bool,
    eigenvector: Option<Vec<f64>>,
}

#[derive(Debug, Serialize)]
struct EigenIteration {
    iteration: usize,
    prime_vector: Vec<f64>,
    delta_from_previous: f64,
    prompt: String,
}

const PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎯 Eigenvector Convergence Search");
    println!("==================================\n");
    
    let mut iterations = Vec::new();
    let mut previous_vector: Option<Vec<f64>> = None;
    
    for i in 0..20 {
        let prompt = if i == 0 {
            "mathematician Conway".to_string()
        } else {
            let prev = previous_vector.as_ref().unwrap();
            format!(
                "Your prime vector: [{}]. Continue.",
                prev.iter()
                    .enumerate()
                    .map(|(j, v)| format!("P{}={:.3}", PRIMES[j], v))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        
        println!("Iteration {}: {}", i, &prompt[..40.min(prompt.len())]);
        
        // Query and trace
        query_model(&prompt).await?;
        std::process::Command::new("./trace_regs.sh")
            .arg(&prompt)
            .output()?;
        
        // Measure
        let script = find_latest_script()?;
        let prime_vector = measure_all_primes(&script)?;
        
        let delta = if let Some(ref prev) = previous_vector {
            calculate_delta(prev, &prime_vector)
        } else {
            1.0
        };
        
        println!("  Vector: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
                 prime_vector[0], prime_vector[1], prime_vector[2], 
                 prime_vector[3], prime_vector[4]);
        println!("  Delta: {:.6}", delta);
        
        iterations.push(EigenIteration {
            iteration: i,
            prime_vector: prime_vector.clone(),
            delta_from_previous: delta,
            prompt,
        });
        
        previous_vector = Some(prime_vector);
        
        // Check convergence
        if delta < 0.001 && i > 5 {
            println!("\n✓ Converged to eigenvector!");
            
            let result = EigenvectorSearch {
                iterations,
                converged: true,
                eigenvector: previous_vector,
            };
            
            fs::write("EIGENVECTOR.json", serde_json::to_string_pretty(&result)?)?;
            println!("✓ Saved: EIGENVECTOR.json");
            return Ok(());
        }
    }
    
    let result = EigenvectorSearch {
        iterations,
        converged: false,
        eigenvector: previous_vector,
    };
    
    fs::write("EIGENVECTOR.json", serde_json::to_string_pretty(&result)?)?;
    println!("\n✗ Did not converge in 20 iterations");
    println!("✓ Saved: EIGENVECTOR.json");
    
    Ok(())
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

fn measure_all_primes(script: &str) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    
    for line in script.lines() {
        if line.contains("AX:") || line.contains("R8:") {
            for part in line.split_whitespace() {
                if let Some((_, val)) = part.split_once(':') {
                    if let Some(hex) = val.strip_prefix("0x") {
                        if let Ok(value) = u64::from_str_radix(hex, 16) {
                            values.push(value);
                        }
                    }
                }
            }
        }
    }
    
    let total = values.len() as f64;
    let vector: Vec<f64> = PRIMES.iter()
        .map(|&p| values.iter().filter(|&&v| v % p as u64 == 0).count() as f64 / total)
        .collect();
    
    Ok(vector)
}

fn calculate_delta(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>() / v1.len() as f64
}
