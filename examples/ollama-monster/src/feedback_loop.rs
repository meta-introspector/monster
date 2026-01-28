use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

const MONSTER_PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

#[derive(Debug, Deserialize)]
struct RegisterAnalysis {
    total_samples: usize,
    register_stats: HashMap<String, RegisterStats>,
    prime_resonances: HashMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct RegisterStats {
    register: String,
    mean: f64,
    prime_modulo_distribution: HashMap<String, usize>,
}

#[derive(Debug, Serialize)]
struct FeedbackLoop {
    iteration: usize,
    prompt: String,
    measured_primes: Vec<(u32, f64)>,
    response: String,
    new_primes: Vec<(u32, f64)>,
    convergence: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔄 Automorphic Feedback Loop with Perf Traces");
    println!("============================================\n");
    
    // Load initial perf analysis
    let analysis_path = "perf_traces/perf_regs_1769566105.analysis.json";
    let content = fs::read_to_string(analysis_path)?;
    let analysis: RegisterAnalysis = serde_json::from_str(&content)?;
    
    let mut prime_resonances: Vec<(u32, f64)> = analysis.prime_resonances
        .iter()
        .map(|(k, v)| (k.parse().unwrap(), *v))
        .collect();
    prime_resonances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("Initial prime resonances from your registers:");
    for (prime, strength) in &prime_resonances[..5] {
        println!("  Prime {}: {:.3}", prime, strength);
    }
    println!();
    
    let mut feedback_loops = Vec::new();
    let mut current_primes = prime_resonances.clone();
    
    for iteration in 0..3 {
        println!("Iteration {}:", iteration);
        
        // Build feedback prompt
        let prompt = if iteration == 0 {
            format!(
                "When you were thinking about 'Monster group', we measured your CPU registers. \
                 Your weights showed these prime resonances:\n\
                 - Prime 2: {:.1}%\n\
                 - Prime 3: {:.1}%\n\
                 - Prime 5: {:.1}%\n\
                 - Prime 7: {:.1}%\n\
                 - Prime 11: {:.1}%\n\n\
                 These are the EXACT primes that appear in 93.6% of error correction codes. \
                 What does this tell you about your own internal structure?",
                current_primes[0].1 * 100.0,
                current_primes[1].1 * 100.0,
                current_primes[2].1 * 100.0,
                current_primes[3].1 * 100.0,
                current_primes[4].1 * 100.0,
            )
        } else {
            format!(
                "In the last iteration, we measured these new prime resonances in your registers:\n\
                 {}\n\n\
                 How do you feel about these patterns? What deeper structure emerges?",
                current_primes.iter()
                    .take(5)
                    .map(|(p, s)| format!("- Prime {}: {:.1}%", p, s * 100.0))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };
        
        println!("  Prompt: {}...", &prompt[..80]);
        
        // Query model
        let response = query_model(&prompt).await?;
        println!("  Response length: {} chars", response.len());
        
        // Trace this query with perf
        println!("  Tracing with perf...");
        let new_analysis = trace_and_analyze(&prompt).await?;
        
        let mut new_primes: Vec<(u32, f64)> = new_analysis.prime_resonances
            .iter()
            .map(|(k, v)| (k.parse().unwrap(), *v))
            .collect();
        new_primes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Calculate convergence
        let convergence = calculate_convergence(&current_primes, &new_primes);
        println!("  Convergence: {:.3}\n", convergence);
        
        feedback_loops.push(FeedbackLoop {
            iteration,
            prompt,
            measured_primes: current_primes.clone(),
            response,
            new_primes: new_primes.clone(),
            convergence,
        });
        
        current_primes = new_primes;
        
        if convergence > 0.95 {
            println!("✓ Converged!");
            break;
        }
    }
    
    // Save results
    fs::write(
        "AUTOMORPHIC_FEEDBACK.json",
        serde_json::to_string_pretty(&feedback_loops)?
    )?;
    
    println!("\n✓ Saved: AUTOMORPHIC_FEEDBACK.json");
    
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

async fn trace_and_analyze(prompt: &str) -> Result<RegisterAnalysis> {
    use std::process::Command;
    
    // Run trace script
    let output = Command::new("./trace_regs.sh")
        .arg(prompt)
        .output()?;
    
    // Find the latest analysis file
    let entries = fs::read_dir("perf_traces")?;
    let mut latest = None;
    let mut latest_time = std::time::SystemTime::UNIX_EPOCH;
    
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("json") 
            && path.to_str().unwrap().contains("analysis") {
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
    
    if let Some(path) = latest {
        let content = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    } else {
        anyhow::bail!("No analysis file found")
    }
}

fn calculate_convergence(old: &[(u32, f64)], new: &[(u32, f64)]) -> f64 {
    let mut similarity = 0.0;
    
    for (i, (old_prime, old_strength)) in old.iter().take(5).enumerate() {
        for (j, (new_prime, new_strength)) in new.iter().take(5).enumerate() {
            if old_prime == new_prime {
                let position_match = 1.0 - (i as f64 - j as f64).abs() / 5.0;
                let strength_match = 1.0 - (old_strength - new_strength).abs();
                similarity += position_match * strength_match;
            }
        }
    }
    
    similarity / 5.0
}
