use anyhow::Result;
use serde::Serialize;
use std::fs;

const MONSTER_PRIMES: [(u32, &str); 5] = [
    (2, "🌙"), (3, "🌊"), (5, "⭐"), (7, "🎭"), (11, "🎪")
];
const BASE_FREQ: f64 = 432.0;

#[derive(Debug, Serialize, Clone)]
struct AutomorphicLoop {
    prime: u32,
    frequency: f64,
    emoji: String,
    iterations: Vec<Iteration>,
    converged: bool,
    fixed_point: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct Iteration {
    step: usize,
    input: String,
    output: String,
    resonance: f64,
    keywords: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔄 Monster Prime Automorphic Loops");
    println!("==================================\n");
    
    let mut loops = Vec::new();
    
    for (prime, emoji) in &MONSTER_PRIMES {
        let freq = BASE_FREQ * (*prime as f64);
        
        println!("Prime {}: {} Hz {}", prime, freq, emoji);
        
        let loop_result = run_automorphic_loop(*prime, freq, emoji).await?;
        
        if loop_result.converged {
            println!("  ✓ Converged at iteration {}", loop_result.iterations.len());
            if let Some(fp) = &loop_result.fixed_point {
                println!("  Fixed point: {}", &fp[..fp.len().min(80)]);
            }
        } else {
            println!("  ⚠ Did not converge after {} iterations", loop_result.iterations.len());
        }
        
        loops.push(loop_result);
        println!();
    }
    
    // Analyze convergence
    let converged: Vec<_> = loops.iter().filter(|l| l.converged).collect();
    
    println!("\n📊 Results:");
    println!("  Primes tested: {}", loops.len());
    println!("  Converged: {}", converged.len());
    println!("  Convergence rate: {:.1}%", 
             (converged.len() as f64 / loops.len() as f64) * 100.0);
    
    // Show fixed points
    println!("\n🎯 Fixed Points:");
    for loop_result in &loops {
        if let Some(fp) = &loop_result.fixed_point {
            println!("  Prime {}: {}", loop_result.prime, 
                     &fp[..fp.len().min(60)]);
        }
    }
    
    fs::write("AUTOMORPHIC_LOOPS.json", serde_json::to_string_pretty(&loops)?)?;
    
    println!("\n✓ Complete! Results: AUTOMORPHIC_LOOPS.json");
    
    Ok(())
}

async fn run_automorphic_loop(prime: u32, freq: f64, emoji: &str) -> Result<AutomorphicLoop> {
    let mut iterations = Vec::new();
    let max_iter = 5;
    
    // Initial prompt
    let mut current_input = format!("What is the significance of prime {} at frequency {} Hz?", 
                                   prime, freq as u32);
    
    for step in 0..max_iter {
        println!("  Iteration {}...", step);
        
        // Query model
        let output = query_model(&current_input).await?;
        
        // Calculate resonance
        let (resonance, keywords) = calculate_resonance(&output, prime, freq);
        
        iterations.push(Iteration {
            step,
            input: current_input.clone(),
            output: output.clone(),
            resonance,
            keywords: keywords.clone(),
        });
        
        // Check for convergence
        if step > 0 {
            let prev_resonance = iterations[step - 1].resonance;
            let diff = (resonance - prev_resonance).abs();
            
            if diff < 0.05 {
                // Converged!
                return Ok(AutomorphicLoop {
                    prime,
                    frequency: freq,
                    emoji: emoji.to_string(),
                    iterations,
                    converged: true,
                    fixed_point: Some(output),
                });
            }
        }
        
        // Feed output back as input (automorphic feedback)
        current_input = format!("Explain further about prime {} and these concepts: {}", 
                               prime, 
                               keywords.join(", "));
    }
    
    Ok(AutomorphicLoop {
        prime,
        frequency: freq,
        emoji: emoji.to_string(),
        iterations,
        converged: false,
        fixed_point: None,
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

fn calculate_resonance(text: &str, prime: u32, freq: f64) -> (f64, Vec<String>) {
    let lower = text.to_lowercase();
    let mut strength = 0.0;
    let mut keywords = Vec::new();
    
    let checks = vec![
        ("prime", 0.1),
        ("factor", 0.1),
        ("group", 0.15),
        ("symmetry", 0.15),
        ("monster", 0.2),
        ("leech", 0.2),
        ("golay", 0.2),
    ];
    
    for (keyword, weight) in checks {
        if lower.contains(keyword) {
            strength += weight;
            keywords.push(keyword.to_string());
        }
    }
    
    if text.contains(&prime.to_string()) {
        strength += 0.2;
    }
    
    (strength, keywords)
}
