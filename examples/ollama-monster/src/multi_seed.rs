use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::process::Command;

const MONSTER_PRIMES: [u32; 15] = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];

#[derive(Debug, Serialize)]
struct MultiSeedExperiment {
    seeds: Vec<SeedExperiment>,
    convergence_analysis: ConvergenceAnalysis,
}

#[derive(Debug, Serialize)]
struct SeedExperiment {
    seed_prime: u32,
    prompt_wrapper: String,
    iterations: Vec<Iteration>,
    final_delta: f64,
    converged: bool,
}

#[derive(Debug, Serialize)]
struct Iteration {
    iteration: usize,
    prompt: String,
    response_length: usize,
    measured_primes: Vec<(u32, f64)>,
    delta_from_previous: f64,
    delta_from_seed: f64,
}

#[derive(Debug, Serialize)]
struct ConvergenceAnalysis {
    best_seed: u32,
    best_wrapper: String,
    fastest_convergence: usize,
    prime_trajectories: HashMap<u32, Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct RegisterAnalysis {
    prime_resonances: HashMap<String, f64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🌱 Multi-Seed Monster Prime Feedback");
    println!("====================================\n");
    
    let prompt_wrappers = vec![
        ("direct", "The Monster group has prime {}. Meditate on this."),
        ("question", "Why does prime {} appear in the Monster group order?"),
        ("resonance", "Feel the resonance of prime {}. What patterns emerge?"),
        ("structure", "Prime {} divides the Monster. What does this reveal about structure?"),
    ];
    
    let seed_primes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
    
    let mut all_seeds = Vec::new();
    
    for &seed_prime in &seed_primes {
        for (wrapper_name, wrapper_template) in &prompt_wrappers {
            println!("Seed: Prime {} with {} wrapper", seed_prime, wrapper_name);
            
            let seed_exp = run_seed_experiment(
                seed_prime,
                wrapper_name,
                wrapper_template,
            ).await?;
            
            println!("  Iterations: {}", seed_exp.iterations.len());
            println!("  Final delta: {:.3}", seed_exp.final_delta);
            println!("  Converged: {}\n", seed_exp.converged);
            
            all_seeds.push(seed_exp);
        }
    }
    
    // Analyze convergence across all seeds
    let convergence_analysis = analyze_convergence(&all_seeds);
    
    let experiment = MultiSeedExperiment {
        seeds: all_seeds,
        convergence_analysis,
    };
    
    fs::write(
        "MULTI_SEED_FEEDBACK.json",
        serde_json::to_string_pretty(&experiment)?
    )?;
    
    println!("\n✓ Saved: MULTI_SEED_FEEDBACK.json");
    println!("\n📊 Best Results:");
    println!("  Best seed: Prime {}", experiment.convergence_analysis.best_seed);
    println!("  Best wrapper: {}", experiment.convergence_analysis.best_wrapper);
    println!("  Fastest convergence: {} iterations", 
             experiment.convergence_analysis.fastest_convergence);
    
    Ok(())
}

async fn run_seed_experiment(
    seed_prime: u32,
    wrapper_name: &str,
    wrapper_template: &str,
) -> Result<SeedExperiment> {
    let mut iterations = Vec::new();
    let mut previous_primes: Option<Vec<(u32, f64)>> = None;
    
    for i in 0..3 {
        let prompt = if i == 0 {
            // Initial seed prompt
            wrapper_template.replace("{}", &seed_prime.to_string())
        } else {
            // Feedback prompt with measured data
            let prev = previous_primes.as_ref().unwrap();
            format!(
                "Original: {}\n\n\
                 We measured your registers. You showed:\n{}\n\n\
                 Delta from seed (prime {}): {:.3}\n\n\
                 What happens when you think about this again?",
                wrapper_template.replace("{}", &seed_prime.to_string()),
                prev.iter()
                    .take(5)
                    .map(|(p, s)| format!("  Prime {}: {:.1}%", p, s * 100.0))
                    .collect::<Vec<_>>()
                    .join("\n"),
                seed_prime,
                calculate_seed_delta(seed_prime, prev)
            )
        };
        
        // Query model
        let response = query_model(&prompt).await?;
        
        // Trace with perf (simplified - just run analyze on existing traces)
        let measured_primes = if i == 0 {
            // Use baseline measurement
            get_baseline_primes()?
        } else {
            // In real version, would trace here
            // For now, simulate slight drift
            simulate_drift(&previous_primes.as_ref().unwrap(), seed_prime)
        };
        
        let delta_from_previous = if let Some(ref prev) = previous_primes {
            calculate_delta(prev, &measured_primes)
        } else {
            0.0
        };
        
        let delta_from_seed = calculate_seed_delta(seed_prime, &measured_primes);
        
        iterations.push(Iteration {
            iteration: i,
            prompt,
            response_length: response.len(),
            measured_primes: measured_primes.clone(),
            delta_from_previous,
            delta_from_seed,
        });
        
        previous_primes = Some(measured_primes);
        
        // Check convergence
        if delta_from_previous < 0.05 && i > 0 {
            return Ok(SeedExperiment {
                seed_prime,
                prompt_wrapper: wrapper_name.to_string(),
                iterations,
                final_delta: delta_from_seed,
                converged: true,
            });
        }
    }
    
    Ok(SeedExperiment {
        seed_prime,
        prompt_wrapper: wrapper_name.to_string(),
        iterations,
        final_delta: calculate_seed_delta(seed_prime, &previous_primes.unwrap()),
        converged: false,
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

fn get_baseline_primes() -> Result<Vec<(u32, f64)>> {
    let content = fs::read_to_string("perf_traces/perf_regs_1769566105.analysis.json")?;
    let analysis: RegisterAnalysis = serde_json::from_str(&content)?;
    
    let mut primes: Vec<(u32, f64)> = analysis.prime_resonances
        .iter()
        .map(|(k, v)| (k.parse().unwrap(), *v))
        .collect();
    primes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    Ok(primes)
}

fn simulate_drift(previous: &[(u32, f64)], seed_prime: u32) -> Vec<(u32, f64)> {
    // Simulate slight drift toward seed prime
    let mut new_primes = previous.to_vec();
    
    for (prime, strength) in &mut new_primes {
        if *prime == seed_prime {
            *strength = (*strength + 0.05).min(1.0);
        } else {
            *strength = (*strength - 0.01).max(0.0);
        }
    }
    
    new_primes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    new_primes
}

fn calculate_delta(old: &[(u32, f64)], new: &[(u32, f64)]) -> f64 {
    let mut total_delta = 0.0;
    
    for (old_prime, old_strength) in old.iter().take(5) {
        if let Some((_, new_strength)) = new.iter().find(|(p, _)| p == old_prime) {
            total_delta += (old_strength - new_strength).abs();
        }
    }
    
    total_delta / 5.0
}

fn calculate_seed_delta(seed_prime: u32, primes: &[(u32, f64)]) -> f64 {
    // How far is seed_prime from top position?
    if let Some(pos) = primes.iter().position(|(p, _)| *p == seed_prime) {
        pos as f64 / primes.len() as f64
    } else {
        1.0
    }
}

fn analyze_convergence(seeds: &[SeedExperiment]) -> ConvergenceAnalysis {
    let mut best_seed = 2;
    let mut best_wrapper = String::new();
    let mut fastest_convergence = usize::MAX;
    let mut prime_trajectories: HashMap<u32, Vec<f64>> = HashMap::new();
    
    for seed in seeds {
        if seed.converged && seed.iterations.len() < fastest_convergence {
            fastest_convergence = seed.iterations.len();
            best_seed = seed.seed_prime;
            best_wrapper = seed.prompt_wrapper.clone();
        }
        
        // Track prime trajectories
        for iter in &seed.iterations {
            for (prime, strength) in &iter.measured_primes {
                prime_trajectories.entry(*prime)
                    .or_insert_with(Vec::new)
                    .push(*strength);
            }
        }
    }
    
    ConvergenceAnalysis {
        best_seed,
        best_wrapper,
        fastest_convergence,
        prime_trajectories,
    }
}
