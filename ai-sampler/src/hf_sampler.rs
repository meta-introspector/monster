use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use anyhow::Result;
use std::fs;

#[derive(Serialize, Deserialize, Debug)]
struct HFRequest {
    inputs: String,
    parameters: HFParameters,
}

#[derive(Serialize, Deserialize, Debug)]
struct HFParameters {
    max_new_tokens: usize,
    temperature: f32,
}

#[derive(Serialize, Deserialize, Debug)]
struct HFResponse {
    generated_text: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎪 Monster Walk AI Sampler - HuggingFace Edition");
    println!("=================================================\n");
    
    let api_token = std::env::var("HF_API_TOKEN")
        .expect("Set HF_API_TOKEN environment variable");
    
    // HuggingFace models to test
    let models = vec![
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "codellama/CodeLlama-7b-hf",
    ];
    
    // Load Monster Walk content
    let content = load_content()?;
    
    for model in models {
        println!("🤖 Testing model: {}", model);
        
        let prompts = vec![
            "What is the Monster Walk discovery?",
            "Explain Bott periodicity in this context.",
            "How many groups were found?",
        ];
        
        for (i, prompt) in prompts.iter().enumerate() {
            println!("\n  Prompt {}: {}", i + 1, prompt);
            
            let full_prompt = format!(
                "Context: {}\n\nQuestion: {}\n\nAnswer:",
                &content[..1000], prompt
            );
            
            match query_huggingface(model, &full_prompt, &api_token).await {
                Ok(response) => {
                    println!("  Response: {}", &response[..150.min(response.len())]);
                    save_hf_response(model, i, prompt, &response)?;
                }
                Err(e) => {
                    println!("  Error: {}", e);
                }
            }
        }
        
        println!();
    }
    
    println!("✓ HuggingFace sampling complete!");
    println!("  Results: ai-samples/huggingface/");
    
    Ok(())
}

async fn query_huggingface(model: &str, prompt: &str, token: &str) -> Result<String> {
    let client = Client::new();
    
    let request = HFRequest {
        inputs: prompt.to_string(),
        parameters: HFParameters {
            max_new_tokens: 200,
            temperature: 0.7,
        },
    };
    
    let url = format!("https://api-inference.huggingface.co/models/{}", model);
    
    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", token))
        .json(&request)
        .send()
        .await?;
    
    let hf_response: Vec<HFResponse> = response.json().await?;
    
    Ok(hf_response.first()
        .map(|r| r.generated_text.clone())
        .unwrap_or_default())
}

fn load_content() -> Result<String> {
    // Load from previous extraction or use sample
    if let Ok(content) = fs::read_to_string("ai-samples/site-content.txt") {
        Ok(content)
    } else {
        Ok(r#"
The Monster Walk is a hierarchical structure in the Monster group's prime factorization.
By systematically removing prime factors, we discover exactly 10 groups that preserve leading digits.
This exhibits Bott periodicity with period 8 and matches the 10-fold way classification.
        "#.to_string())
    }
}

fn save_hf_response(model: &str, prompt_num: usize, prompt: &str, response: &str) -> Result<()> {
    fs::create_dir_all("ai-samples/huggingface")?;
    
    let model_name = model.split('/').last().unwrap_or(model);
    let filename = format!("ai-samples/huggingface/{}_prompt_{}.json", model_name, prompt_num);
    
    let data = json!({
        "model": model,
        "prompt": prompt,
        "response": response,
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    
    fs::write(filename, serde_json::to_string_pretty(&data)?)?;
    
    Ok(())
}
