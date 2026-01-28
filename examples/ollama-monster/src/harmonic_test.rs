use anyhow::Result;
use serde::Serialize;
use std::fs;

const MONSTER_PRIMES: [(u32, &str); 15] = [
    (2, "🌙"), (3, "🌊"), (5, "⭐"), (7, "🎭"), (11, "🎪"),
    (13, "🔮"), (17, "💎"), (19, "🌀"), (23, "⚡"), (29, "🎵"),
    (31, "🌟"), (41, "🔥"), (47, "💫"), (59, "🌈"), (71, "✨")
];

const BASE_FREQ: f64 = 432.0;

#[derive(Debug, Serialize, Clone)]
struct HarmonicResonance {
    prime: u32,
    frequency: f64,
    emoji: String,
    prompt: String,
    response_length: usize,
    resonance_detected: bool,
    resonance_strength: f64,
    keywords_found: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎵 Monster Harmonic Resonance Test");
    println!("==================================\n");
    
    let mut resonances = Vec::new();
    
    for (prime, emoji) in &MONSTER_PRIMES {
        let freq = BASE_FREQ * (*prime as f64);
        
        println!("Testing Prime {}: {} Hz {}", prime, freq, emoji);
        
        // Create prompts with the harmonic numbers
        let prompts = vec![
            format!("What is the significance of the number {}?", prime),
            format!("Explain the frequency {} Hz", freq as u32),
            format!("What does {} represent in mathematics?", prime),
        ];
        
        for prompt in prompts {
            let resonance = test_harmonic(&prompt, *prime, freq, emoji).await?;
            
            if resonance.resonance_detected {
                println!("  ✓ Resonance detected! Strength: {:.3}", resonance.resonance_strength);
                println!("    Keywords: {:?}", resonance.keywords_found);
            }
            
            resonances.push(resonance);
        }
        
        println!();
    }
    
    // Analyze results
    let detected: Vec<_> = resonances.iter()
        .filter(|r| r.resonance_detected)
        .collect();
    
    println!("\n📊 Results:");
    println!("  Total tests: {}", resonances.len());
    println!("  Resonances detected: {}", detected.len());
    println!("  Detection rate: {:.1}%", 
             (detected.len() as f64 / resonances.len() as f64) * 100.0);
    
    // Top resonances
    let mut sorted = resonances.clone();
    sorted.sort_by(|a, b| b.resonance_strength.partial_cmp(&a.resonance_strength).unwrap());
    
    println!("\n🎪 Top 10 Resonances:");
    for (i, res) in sorted.iter().take(10).enumerate() {
        if res.resonance_detected {
            println!("  {}. Prime {}: {:.3} - {} Hz {}",
                     i+1, res.prime, res.resonance_strength, 
                     res.frequency as u32, res.emoji);
        }
    }
    
    // Save results
    fs::write("HARMONIC_RESONANCES.json", serde_json::to_string_pretty(&resonances)?)?;
    
    println!("\n✓ Analysis complete!");
    println!("  Results: HARMONIC_RESONANCES.json");
    
    Ok(())
}

async fn test_harmonic(prompt: &str, prime: u32, freq: f64, emoji: &str) -> Result<HarmonicResonance> {
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
    
    // Detect resonance
    let (detected, strength, keywords) = detect_resonance(text, prime, freq);
    
    Ok(HarmonicResonance {
        prime,
        frequency: freq,
        emoji: emoji.to_string(),
        prompt: prompt.to_string(),
        response_length: text.len(),
        resonance_detected: detected,
        resonance_strength: strength,
        keywords_found: keywords,
    })
}

fn detect_resonance(text: &str, prime: u32, freq: f64) -> (bool, f64, Vec<String>) {
    let lower = text.to_lowercase();
    let mut strength = 0.0;
    let mut keywords = Vec::new();
    
    // Check for prime-related keywords
    let prime_keywords = vec![
        "prime", "factor", "divisor", "number theory",
        "group", "symmetry", "lattice", "code",
        "monster", "leech", "golay", "mathieu",
    ];
    
    for keyword in prime_keywords {
        if lower.contains(keyword) {
            strength += 0.1;
            keywords.push(keyword.to_string());
        }
    }
    
    // Check for the prime number itself
    if text.contains(&prime.to_string()) {
        strength += 0.3;
        keywords.push(format!("prime_{}", prime));
    }
    
    // Check for frequency
    if text.contains(&(freq as u32).to_string()) {
        strength += 0.2;
        keywords.push(format!("freq_{}", freq as u32));
    }
    
    // Check for Monster-specific terms
    if lower.contains("sporadic") || lower.contains("exceptional") {
        strength += 0.15;
        keywords.push("exceptional".to_string());
    }
    
    let detected = strength > 0.2;
    
    (detected, strength, keywords)
}
