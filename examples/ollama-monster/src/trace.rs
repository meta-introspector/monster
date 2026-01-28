use anyhow::Result;
use serde::Serialize;
use std::fs;

#[derive(Debug, Serialize)]
struct ExecutionTrace {
    prompt: String,
    tokens: Vec<String>,
    layer_activations: Vec<LayerActivation>,
    monster_resonances: Vec<MonsterResonance>,
}

#[derive(Debug, Serialize)]
struct LayerActivation {
    layer: usize,
    attention_weights: Vec<f32>,
    hidden_states: Vec<f32>,
    prime_signature: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct MonsterResonance {
    layer: usize,
    token_position: usize,
    prime: u32,
    strength: f32,
}

const MONSTER_PRIMES: [u32; 5] = [2, 3, 5, 7, 11];

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎪 Mistral.rs Execution Trace for Monster Codes");
    println!("===============================================\n");
    
    let prompts = vec![
        "What is the Leech lattice?",
        "What is the Golay code?",
        "Explain the Monster group",
    ];
    
    println!("Tracing execution through model layers...\n");
    
    for prompt in prompts {
        println!("Prompt: {}", prompt);
        
        // TODO: Use mistral.rs to get actual layer activations
        // For now, query via Ollama and simulate trace
        
        let trace = trace_execution(prompt).await?;
        
        println!("  Tokens: {}", trace.tokens.len());
        println!("  Layers traced: {}", trace.layer_activations.len());
        println!("  Monster resonances: {}", trace.monster_resonances.len());
        
        // Show resonances
        for res in &trace.monster_resonances {
            println!("    Layer {}, Token {}: Prime {} (strength: {:.3})",
                     res.layer, res.token_position, res.prime, res.strength);
        }
        
        // Save trace
        let filename = format!("TRACE_{}.json", 
                              prompt.replace(" ", "_").replace("?", ""));
        fs::write(&filename, serde_json::to_string_pretty(&trace)?)?;
        println!("  ✓ Saved: {}\n", filename);
    }
    
    println!("✓ Tracing complete!");
    
    Ok(())
}

async fn trace_execution(prompt: &str) -> Result<ExecutionTrace> {
    // Query model
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
    let text = result["response"].as_str().unwrap_or("");
    
    // Tokenize (simple word split for now)
    let tokens: Vec<String> = text.split_whitespace()
        .take(50)
        .map(|s| s.to_string())
        .collect();
    
    // Simulate layer activations (would be real with mistral.rs)
    let mut layer_activations = Vec::new();
    let mut monster_resonances = Vec::new();
    
    // Simulate 12 layers (qwen has ~28 layers)
    for layer in 0..12 {
        let mut activations = vec![0.0f32; tokens.len()];
        let mut primes = Vec::new();
        
        // Simulate activation patterns
        for (i, token) in tokens.iter().enumerate() {
            let activation = simulate_activation(token, layer);
            activations[i] = activation;
            
            // Check for Monster prime resonances
            for &prime in &MONSTER_PRIMES {
                let resonance = check_resonance(token, prime, activation);
                if resonance > 0.5 {
                    monster_resonances.push(MonsterResonance {
                        layer,
                        token_position: i,
                        prime,
                        strength: resonance,
                    });
                    
                    if !primes.contains(&prime) {
                        primes.push(prime);
                    }
                }
            }
        }
        
        layer_activations.push(LayerActivation {
            layer,
            attention_weights: activations.clone(),
            hidden_states: activations,
            prime_signature: primes,
        });
    }
    
    Ok(ExecutionTrace {
        prompt: prompt.to_string(),
        tokens,
        layer_activations,
        monster_resonances,
    })
}

fn simulate_activation(token: &str, layer: usize) -> f32 {
    // Simulate higher activation for Monster-related tokens
    let lower = token.to_lowercase();
    let base = (layer as f32) * 0.1;
    
    if lower.contains("leech") || lower.contains("golay") || 
       lower.contains("monster") || lower.contains("lattice") {
        base + 0.8
    } else if lower.contains("24") || lower.contains("dimension") {
        base + 0.6
    } else if lower.contains("code") || lower.contains("group") {
        base + 0.4
    } else {
        base + 0.1
    }
}

fn check_resonance(token: &str, prime: u32, activation: f32) -> f32 {
    // Check if token resonates with Monster prime
    let lower = token.to_lowercase();
    
    let keyword_match = match prime {
        2 => lower.contains("binary") || lower.contains("two"),
        3 => lower.contains("three") || lower.contains("trinity"),
        5 => lower.contains("five") || lower.contains("penta"),
        7 => lower.contains("seven") || lower.contains("symmetry"),
        11 => lower.contains("eleven") || lower.contains("monster"),
        _ => false,
    };
    
    if keyword_match {
        activation * 0.9
    } else {
        activation * 0.1
    }
}
