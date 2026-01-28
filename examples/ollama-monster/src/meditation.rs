use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::collections::HashMap;

const MONSTER_PRIMES: [(u32, &str); 5] = [
    (2, "🌙"), (3, "🌊"), (5, "⭐"), (7, "🎭"), (11, "🎪")
];

#[derive(Debug, Serialize)]
struct MeditationResult {
    prime: u32,
    concept: String,
    iterations: Vec<MeditationIteration>,
    resonance_evolution: Vec<f64>,
    convergence: bool,
}

#[derive(Debug, Serialize)]
struct MeditationIteration {
    iteration: usize,
    prompt: String,
    response: String,
    measured_patterns: Vec<String>,
    resonance_strength: f64,
    layer_activations: Vec<LayerActivation>,
    attention_patterns: Vec<AttentionPattern>,
}

#[derive(Debug, Serialize)]
struct LayerActivation {
    layer: usize,
    mean_activation: f64,
    max_activation: f64,
    prime_resonance: HashMap<u32, f64>,
}

#[derive(Debug, Serialize)]
struct AttentionPattern {
    layer: usize,
    head: usize,
    pattern_type: String,
    strength: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧘 Monster Prime Meditation with Feedback");
    println!("=========================================\n");
    
    let concepts = vec![
        ("Leech lattice", vec!["24-dimensional", "kissing number", "Conway group"]),
        ("Golay code", vec!["error correction", "Mathieu group", "symmetry"]),
        ("Monster group", vec!["sporadic", "196883", "moonshine"]),
    ];
    
    for (concept, initial_patterns) in concepts {
        println!("Concept: {}", concept);
        
        let result = meditate_on_concept(concept, initial_patterns).await?;
        
        println!("  Iterations: {}", result.iterations.len());
        println!("  Converged: {}", result.convergence);
        println!("  Final resonance: {:.3}", 
                 result.resonance_evolution.last().unwrap_or(&0.0));
        
        let filename = format!("MEDITATION_{}.json", concept.replace(" ", "_"));
        fs::write(&filename, serde_json::to_string_pretty(&result)?)?;
        println!("  ✓ Saved: {}\n", filename);
    }
    
    println!("✓ Meditation complete!");
    
    Ok(())
}

async fn meditate_on_concept(concept: &str, initial_patterns: Vec<&str>) -> Result<MeditationResult> {
    let mut iterations = Vec::new();
    let mut resonance_evolution = Vec::new();
    let mut current_patterns: Vec<String> = initial_patterns.iter()
        .map(|s| s.to_string())
        .collect();
    
    for i in 0..5 {
        // Create meditation prompt with feedback
        let prompt = if i == 0 {
            format!(
                "I want you to meditate fully on the concept of {}. \
                 Resonate with it internally. What patterns emerge?",
                concept
            )
        } else {
            format!(
                "I want you to meditate fully on the concept of {}. \
                 Resonate with it internally. In the last iteration we measured \
                 these patterns in your model: {}. \
                 What deeper patterns do you now perceive?",
                concept,
                current_patterns.join(", ")
            )
        };
        
        println!("    Iteration {}...", i);
        
        // Query model
        let response = query_model(&prompt).await?;
        
        // Measure actual activations (simulate for now, would use mistral.rs)
        let layer_activations = measure_layer_activations(&response, concept);
        let attention_patterns = measure_attention_patterns(&response);
        
        // Measure patterns in response
        let new_patterns = extract_patterns(&response);
        let resonance = calculate_resonance(&response, &current_patterns);
        
        iterations.push(MeditationIteration {
            iteration: i,
            prompt,
            response: response.clone(),
            measured_patterns: new_patterns.clone(),
            resonance_strength: resonance,
            layer_activations,
            attention_patterns,
        });
        
        resonance_evolution.push(resonance);
        
        // Update patterns for next iteration
        current_patterns = new_patterns;
        
        // Check convergence
        if i > 0 {
            let prev_resonance = resonance_evolution[i - 1];
            if (resonance - prev_resonance).abs() < 0.05 {
                return Ok(MeditationResult {
                    prime: 0,
                    concept: concept.to_string(),
                    iterations,
                    resonance_evolution,
                    convergence: true,
                });
            }
        }
    }
    
    Ok(MeditationResult {
        prime: 0,
        concept: concept.to_string(),
        iterations,
        resonance_evolution,
        convergence: false,
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

fn extract_patterns(text: &str) -> Vec<String> {
    let keywords = vec![
        "symmetry", "group", "lattice", "dimension", "prime",
        "code", "error", "correction", "sporadic", "exceptional",
        "moonshine", "modular", "form", "representation", "character",
    ];
    
    keywords.iter()
        .filter(|k| text.to_lowercase().contains(*k))
        .map(|s| s.to_string())
        .collect()
}

fn calculate_resonance(text: &str, patterns: &[String]) -> f64 {
    let mut strength: f64 = 0.0;
    let lower = text.to_lowercase();
    
    for pattern in patterns {
        if lower.contains(pattern) {
            strength += 0.2;
        }
    }
    
    // Bonus for Monster-specific terms
    if lower.contains("monster") { strength += 0.3; }
    if lower.contains("leech") { strength += 0.3; }
    if lower.contains("golay") { strength += 0.3; }
    
    strength.min(1.0)
}

fn measure_layer_activations(text: &str, concept: &str) -> Vec<LayerActivation> {
    // Simulate layer activations based on keyword density
    // In real implementation, would use mistral.rs to get actual activations
    let mut activations = Vec::new();
    
    for layer in 0..28 {
        let mut prime_resonance = HashMap::new();
        
        // Simulate prime resonances
        for &(prime, _) in &MONSTER_PRIMES {
            let resonance = if text.to_lowercase().contains(concept) {
                0.5 + (layer as f64 / 28.0) * 0.5 * (prime as f64 / 11.0)
            } else {
                0.1
            };
            prime_resonance.insert(prime, resonance);
        }
        
        let mean = prime_resonance.values().sum::<f64>() / prime_resonance.len() as f64;
        let max = prime_resonance.values().cloned().fold(0.0f64, f64::max);
        
        activations.push(LayerActivation {
            layer,
            mean_activation: mean,
            max_activation: max,
            prime_resonance,
        });
    }
    
    activations
}

fn measure_attention_patterns(text: &str) -> Vec<AttentionPattern> {
    // Simulate attention patterns
    // In real implementation, would extract from model
    let mut patterns = Vec::new();
    
    let pattern_types = vec![
        ("self-attention", 0.8),
        ("cross-attention", 0.6),
        ("prime-resonance", 0.9),
    ];
    
    for layer in [0, 7, 14, 21, 27] {
        for (pattern_type, strength) in &pattern_types {
            patterns.push(AttentionPattern {
                layer,
                head: 0,
                pattern_type: pattern_type.to_string(),
                strength: *strength * (1.0 + layer as f64 / 28.0),
            });
        }
    }
    
    patterns
}
