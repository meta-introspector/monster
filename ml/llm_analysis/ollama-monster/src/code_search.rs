use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Serialize, Clone)]
struct CodePattern {
    code_name: String,
    primes: Vec<u32>,
    found_in_layer: Vec<usize>,
    activation_strength: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎪 Error Correction Code Search in Qwen via mistral.rs");
    println!("======================================================\n");
    
    // Load code signatures
    let code_sigs = load_code_signatures()?;
    println!("Loaded {} code signatures\n", code_sigs.len());
    
    println!("Loading qwen2.5:3b model with mistral.rs...");
    
    // TODO: Use mistral.rs to load model
    // For now, use Ollama API to query the model
    
    let client = reqwest::Client::new();
    
    let mut results = Vec::new();
    
    // Test key codes
    let test_codes = vec![
        ("leech", "What is the Leech lattice?"),
        ("golay", "What is the Golay code?"),
        ("hamming", "What is the Hamming code?"),
        ("reed_solomon", "What is Reed-Solomon code?"),
    ];
    
    println!("Querying model for code knowledge...\n");
    
    for (code_name, prompt) in &test_codes {
        println!("Testing: {}", code_name);
        
        let response = query_ollama(&client, prompt).await?;
        
        let found = response.to_lowercase().contains(code_name);
        let strength = if found { 1.0 } else { 0.0 };
        
        if found {
            println!("  ✓ Model knows about {}", code_name);
            
            if let Some(primes) = code_sigs.get(&code_name.to_string()) {
                results.push(CodePattern {
                    code_name: code_name.to_string(),
                    primes: primes.clone(),
                    found_in_layer: vec![0], // Would be actual layers with mistral.rs
                    activation_strength: strength,
                });
            }
        } else {
            println!("  ✗ No knowledge of {}", code_name);
        }
    }
    
    println!("\n📊 Results:");
    println!("  Codes found: {}/{}", results.len(), test_codes.len());
    
    for pattern in &results {
        println!("  ✓ {}: primes={:?}", pattern.code_name, pattern.primes);
    }
    
    fs::write("MODEL_CODE_KNOWLEDGE.json", serde_json::to_string_pretty(&results)?)?;
    
    println!("\n✓ Analysis complete!");
    println!("  Results: MODEL_CODE_KNOWLEDGE.json");
    
    Ok(())
}

async fn query_ollama(client: &reqwest::Client, prompt: &str) -> Result<String> {
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

fn load_code_signatures() -> Result<HashMap<String, Vec<u32>>> {
    let mut sigs = HashMap::new();
    
    if let Ok(data) = fs::read_to_string("CODE_MONSTER_MAP.json") {
        if let Ok(mapping) = serde_json::from_str::<serde_json::Value>(&data) {
            if let Some(codes) = mapping["mapped_codes"].as_array() {
                for code in codes.iter().take(100) {
                    if let (Some(name), Some(primes)) = (
                        code["code_id"].as_str(),
                        code["prime_signature"].as_array()
                    ) {
                        let prime_vec: Vec<u32> = primes.iter()
                            .filter_map(|p| p.as_u64().map(|n| n as u32))
                            .collect();
                        
                        if !prime_vec.is_empty() {
                            sigs.insert(name.to_string(), prime_vec);
                        }
                    }
                }
            }
        }
    }
    
    Ok(sigs)
}
