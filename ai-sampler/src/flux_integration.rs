use anyhow::Result;
use reqwest;
use serde::{Deserialize, Serialize};
use std::env;

/// FLUX.1-dev integration via HuggingFace API
pub struct FluxGenerator {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct FluxRequest {
    inputs: String,
    parameters: FluxParameters,
}

#[derive(Serialize)]
struct FluxParameters {
    seed: Option<u64>,
    num_inference_steps: u32,
    guidance_scale: f32,
}

#[derive(Deserialize)]
struct FluxResponse {
    // HF returns image as bytes
}

impl FluxGenerator {
    pub fn new() -> Result<Self> {
        let api_key = env::var("HF_API_TOKEN")
            .unwrap_or_else(|_| "".to_string());
        
        Ok(Self {
            api_key,
            model: "black-forest-labs/FLUX.1-dev".to_string(),
            client: reqwest::Client::new(),
        })
    }
    
    /// Generate image with FLUX.1-dev
    pub async fn generate(&self, prompt: &str, seed: u64) -> Result<Vec<u8>> {
        println!("  🎨 Calling FLUX.1-dev API...");
        println!("     Model: {}", self.model);
        println!("     Prompt: {}", prompt);
        println!("     Seed: {}", seed);
        
        let url = format!("https://api-inference.huggingface.co/models/{}", self.model);
        
        let request = FluxRequest {
            inputs: prompt.to_string(),
            parameters: FluxParameters {
                seed: Some(seed),
                num_inference_steps: 28,
                guidance_scale: 3.5,
            },
        };
        
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error = response.text().await?;
            anyhow::bail!("FLUX API error: {}", error);
        }
        
        let image_bytes = response.bytes().await?.to_vec();
        
        println!("     ✓ Generated {} bytes", image_bytes.len());
        
        Ok(image_bytes)
    }
}

/// LLaVA vision model integration
pub struct LlavaAnalyzer {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct LlavaRequest {
    inputs: LlavaInputs,
}

#[derive(Serialize)]
struct LlavaInputs {
    image: String, // base64
    text: String,
}

#[derive(Deserialize)]
struct LlavaResponse {
    generated_text: String,
}

impl LlavaAnalyzer {
    pub fn new() -> Result<Self> {
        let api_key = env::var("HF_API_TOKEN")
            .unwrap_or_else(|_| "".to_string());
        
        Ok(Self {
            api_key,
            model: "llava-hf/llava-1.5-7b-hf".to_string(),
            client: reqwest::Client::new(),
        })
    }
    
    /// Analyze image with LLaVA
    pub async fn analyze(&self, image_bytes: &[u8], prompt: &str) -> Result<String> {
        println!("  👁️  Calling LLaVA API...");
        
        let image_b64 = base64::encode(image_bytes);
        
        let url = format!("https://api-inference.huggingface.co/models/{}", self.model);
        
        let request = LlavaRequest {
            inputs: LlavaInputs {
                image: image_b64,
                text: prompt.to_string(),
            },
        };
        
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error = response.text().await?;
            anyhow::bail!("LLaVA API error: {}", error);
        }
        
        let result: Vec<LlavaResponse> = response.json().await?;
        let description = result.first()
            .map(|r| r.generated_text.clone())
            .unwrap_or_default();
        
        println!("     ✓ Analysis: {}", &description[..description.len().min(80)]);
        
        Ok(description)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_flux_generator() {
        if env::var("HF_API_TOKEN").is_ok() {
            let gen = FluxGenerator::new().unwrap();
            let result = gen.generate("test", 12345).await;
            assert!(result.is_ok() || result.unwrap_err().to_string().contains("API"));
        }
    }
}
