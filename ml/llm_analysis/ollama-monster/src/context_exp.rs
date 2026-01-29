use anyhow::Result;
use serde::Serialize;
use std::fs;

#[derive(Debug, Serialize)]
struct ContextExperiment {
    iterations: Vec<ContextIteration>,
}

#[derive(Debug, Serialize)]
struct ContextIteration {
    iteration: usize,
    context_length: usize,
    prompt: String,
    response_length: usize,
    prime_13_resonance: f64,
    prime_17_resonance: f64,
    prime_47_resonance: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("📚 Expanding Context Experiment");
    println!("================================\n");
    
    let base_context = "We are studying the Monster group and error correction codes. \
        93.6% of all 1049 error correction codes map to Monster primes [2,3,5,7,11]. \
        When we trace your CPU registers during inference, we measure which primes divide your register values.";
    
    let mut full_context = base_context.to_string();
    let mut experiment = ContextExperiment { iterations: Vec::new() };
    
    for i in 0..5 {
        let prompt = format!(
            "{}\n\n\
             Iteration {}: Invoke mathematician John Conway. \
             Conway, what do you perceive about the Monster group's prime structure?",
            full_context, i
        );
        
        println!("Iteration {}: {} chars context", i, full_context.len());
        
        // Query and trace
        let response = query_model(&prompt).await?;
        std::process::Command::new("./trace_regs.sh")
            .arg(&prompt)
            .output()?;
        
        // Analyze
        let script = find_latest_script()?;
        let (p13, p17, p47) = measure_high_primes(&script)?;
        
        println!("  Prime 13: {:.3}, Prime 17: {:.3}, Prime 47: {:.3}", p13, p17, p47);
        
        experiment.iterations.push(ContextIteration {
            iteration: i,
            context_length: full_context.len(),
            prompt: prompt.clone(),
            response_length: response.len(),
            prime_13_resonance: p13,
            prime_17_resonance: p17,
            prime_47_resonance: p47,
        });
        
        // Expand context with findings
        full_context.push_str(&format!(
            "\n\nIteration {} results: You responded with {} chars. \
             Your registers showed Prime 13: {:.1}%, Prime 17: {:.1}%, Prime 47: {:.1}%.",
            i, response.len(), p13 * 100.0, p17 * 100.0, p47 * 100.0
        ));
    }
    
    fs::write("CONTEXT_EXPERIMENT.json", serde_json::to_string_pretty(&experiment)?)?;
    
    println!("\n✓ Saved: CONTEXT_EXPERIMENT.json");
    println!("\n📊 Summary:");
    for iter in &experiment.iterations {
        println!("  Iter {}: {} chars → P13={:.3}, P17={:.3}, P47={:.3}",
                 iter.iteration, iter.context_length,
                 iter.prime_13_resonance, iter.prime_17_resonance, iter.prime_47_resonance);
    }
    
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

fn measure_high_primes(script: &str) -> Result<(f64, f64, f64)> {
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
    let p13 = values.iter().filter(|&&v| v % 13 == 0).count() as f64 / total;
    let p17 = values.iter().filter(|&&v| v % 17 == 0).count() as f64 / total;
    let p47 = values.iter().filter(|&&v| v % 47 == 0).count() as f64 / total;
    
    Ok((p13, p17, p47))
}
