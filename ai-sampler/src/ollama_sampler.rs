use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use anyhow::Result;
use headless_chrome::{Browser, LaunchOptions};
use std::fs;

#[derive(Serialize, Deserialize, Debug)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct OllamaResponse {
    response: String,
    done: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎪 Monster Walk AI Sampler - Ollama Edition");
    println!("============================================\n");
    
    // Step 1: Extract content from site
    println!("📖 Extracting content from Monster Walk site...");
    let content = extract_site_content()?;
    println!("✓ Extracted {} characters\n", content.len());
    
    // Step 2: Sample with Ollama models
    let models = vec!["llama2", "mistral", "codellama"];
    
    for model in models {
        println!("🤖 Sampling with {}...", model);
        
        let prompts = vec![
            "Summarize the Monster Walk discovery in 3 sentences.",
            "What is Bott periodicity and how does it relate to the Monster group?",
            "Explain the 10-fold way connection.",
            "What are the harmonic frequencies of the first group?",
        ];
        
        for (i, prompt) in prompts.iter().enumerate() {
            println!("\n  Prompt {}: {}", i + 1, prompt);
            
            match query_ollama(model, &format!("{}\n\nContext:\n{}", prompt, &content[..2000])).await {
                Ok(response) => {
                    println!("  Response: {}", &response[..200.min(response.len())]);
                    
                    // Save response
                    save_response(model, i, prompt, &response)?;
                }
                Err(e) => {
                    println!("  Error: {}", e);
                }
            }
        }
        
        println!();
    }
    
    // Step 3: Generate report
    println!("📊 Generating AI sampling report...");
    generate_report()?;
    
    println!("\n✓ Complete! Results saved to ai-samples/");
    
    Ok(())
}

fn extract_site_content() -> Result<String> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    // Load the Pyodide site
    let html_path = std::env::current_dir()?.join("result/index.html");
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    std::thread::sleep(std::time::Duration::from_secs(3));
    
    // Extract structured content
    let content = tab.evaluate(r#"
        JSON.stringify({
            title: document.title,
            intro: document.querySelector('#intro')?.textContent || '',
            groups: Array.from(document.querySelectorAll('.group-card')).map(card => 
                card.textContent.trim()
            ).join('\n'),
            sections: Array.from(document.querySelectorAll('section')).map(s => ({
                id: s.id,
                text: s.textContent.substring(0, 500)
            }))
        })
    "#, false)?;
    
    if let Some(serde_json::Value::String(json_str)) = content.value {
        let data: serde_json::Value = serde_json::from_str(&json_str)?;
        Ok(serde_json::to_string_pretty(&data)?)
    } else {
        Ok(String::new())
    }
}

async fn query_ollama(model: &str, prompt: &str) -> Result<String> {
    let client = Client::new();
    
    let request = OllamaRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        stream: false,
    };
    
    let response = client
        .post("http://localhost:11434/api/generate")
        .json(&request)
        .send()
        .await?;
    
    let ollama_response: OllamaResponse = response.json().await?;
    
    Ok(ollama_response.response)
}

fn save_response(model: &str, prompt_num: usize, prompt: &str, response: &str) -> Result<()> {
    fs::create_dir_all("ai-samples/ollama")?;
    
    let filename = format!("ai-samples/ollama/{}_prompt_{}.json", model, prompt_num);
    
    let data = json!({
        "model": model,
        "prompt": prompt,
        "response": response,
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    
    fs::write(filename, serde_json::to_string_pretty(&data)?)?;
    
    Ok(())
}

fn generate_report() -> Result<()> {
    let mut report = String::from("# Monster Walk AI Sampling Report\n\n");
    report.push_str("## Ollama Models\n\n");
    
    // Read all samples
    for entry in fs::read_dir("ai-samples/ollama")? {
        let entry = entry?;
        let content = fs::read_to_string(entry.path())?;
        let data: serde_json::Value = serde_json::from_str(&content)?;
        
        report.push_str(&format!("### {}\n", data["model"].as_str().unwrap()));
        report.push_str(&format!("**Prompt**: {}\n\n", data["prompt"].as_str().unwrap()));
        report.push_str(&format!("**Response**: {}\n\n", data["response"].as_str().unwrap()));
        report.push_str("---\n\n");
    }
    
    fs::write("ai-samples/REPORT.md", report)?;
    
    Ok(())
}
