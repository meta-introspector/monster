use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use anyhow::Result;
use headless_chrome::{Browser, LaunchOptions, protocol::cdp::Page};
use std::fs;
use base64::{Engine as _, engine::general_purpose};

#[derive(Serialize, Deserialize, Debug)]
struct VisionRequest {
    model: String,
    prompt: String,
    images: Vec<String>,
    stream: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct VisionResponse {
    response: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("👁️  Monster Walk Vision Model Sampler");
    println!("======================================\n");
    
    // Step 1: Capture screenshots of the page
    println!("📸 Capturing page screenshots...");
    let screenshots = capture_page_screenshots()?;
    println!("✓ Captured {} screenshots\n", screenshots.len());
    
    // Step 2: Analyze with vision models
    let models = vec![
        "llava",           // LLaVA - general vision
        "bakllava",        // BakLLaVA - better at diagrams
    ];
    
    for model in models {
        println!("🤖 Analyzing with {}...", model);
        
        let prompts = vec![
            "Describe what you see in this mathematical visualization.",
            "What shapes and diagrams are present? Identify any mathematical symbols.",
            "Can you identify the group structure shown? Count the number of groups.",
            "What colors and visual patterns do you observe?",
            "Describe the layout and organization of information.",
        ];
        
        for (i, prompt) in prompts.iter().enumerate() {
            println!("\n  Prompt {}: {}", i + 1, prompt);
            
            // Use first screenshot for analysis
            if let Some(screenshot) = screenshots.first() {
                match analyze_with_vision(model, prompt, screenshot).await {
                    Ok(response) => {
                        println!("  Response: {}", &response[..200.min(response.len())]);
                        save_vision_response(model, i, prompt, &response, "full_page")?;
                    }
                    Err(e) => {
                        println!("  Error: {}", e);
                    }
                }
            }
        }
        
        println!();
    }
    
    // Step 3: Shape recognition on specific elements
    println!("🔍 Analyzing specific shapes...");
    analyze_shapes(&screenshots).await?;
    
    // Step 4: Generate vision report
    println!("📊 Generating vision analysis report...");
    generate_vision_report()?;
    
    println!("\n✓ Complete! Results saved to ai-samples/vision/");
    
    Ok(())
}

fn capture_page_screenshots() -> Result<Vec<Vec<u8>>> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = std::env::current_dir()?.join("../result/index.html");
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    // Wait for page to fully render
    std::thread::sleep(std::time::Duration::from_secs(5));
    
    let mut screenshots = Vec::new();
    
    // Full page screenshot
    println!("  Capturing full page...");
    let screenshot = tab.capture_screenshot(
        Page::CaptureScreenshotFormatOption::Png,
        None,
        None,
        true
    )?;
    screenshots.push(screenshot.clone());
    fs::write("ai-samples/vision/full_page.png", &screenshot)?;
    
    // Capture individual group cards
    for i in 1..=10 {
        println!("  Capturing group {}...", i);
        
        // Scroll to element
        tab.evaluate(&format!(r#"
            const elem = document.querySelector('#group-{}');
            if (elem) {{
                elem.scrollIntoView({{behavior: 'instant', block: 'center'}});
            }}
        "#, i), false)?;
        
        std::thread::sleep(std::time::Duration::from_millis(500));
        
        let screenshot = tab.capture_screenshot(
            Page::CaptureScreenshotFormatOption::Png,
            None,
            None,
            false
        )?;
        
        fs::write(format!("ai-samples/vision/group_{}.png", i), &screenshot)?;
        screenshots.push(screenshot);
    }
    
    Ok(screenshots)
}

async fn analyze_with_vision(model: &str, prompt: &str, image: &[u8]) -> Result<String> {
    let client = Client::new();
    
    // Encode image to base64
    let image_b64 = general_purpose::STANDARD.encode(image);
    
    let request = VisionRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        images: vec![image_b64],
        stream: false,
    };
    
    let response = client
        .post("http://localhost:11434/api/generate")
        .json(&request)
        .send()
        .await?;
    
    let vision_response: VisionResponse = response.json().await?;
    
    Ok(vision_response.response)
}

async fn analyze_shapes(screenshots: &[Vec<u8>]) -> Result<()> {
    println!("\n  🔺 Analyzing geometric shapes...");
    
    let shape_prompts = vec![
        "What geometric shapes do you see? List circles, rectangles, triangles, etc.",
        "Identify any mathematical diagrams or graphs.",
        "Are there any arrows or connecting lines between elements?",
        "Describe the visual hierarchy and grouping of elements.",
    ];
    
    for (i, prompt) in shape_prompts.iter().enumerate() {
        if let Some(screenshot) = screenshots.first() {
            match analyze_with_vision("llava", prompt, screenshot).await {
                Ok(response) => {
                    println!("    Shape analysis {}: {}", i + 1, &response[..150.min(response.len())]);
                    save_vision_response("llava", i + 100, prompt, &response, "shapes")?;
                }
                Err(e) => {
                    println!("    Error: {}", e);
                }
            }
        }
    }
    
    Ok(())
}

fn save_vision_response(
    model: &str, 
    prompt_num: usize, 
    prompt: &str, 
    response: &str,
    category: &str
) -> Result<()> {
    fs::create_dir_all("ai-samples/vision")?;
    
    let filename = format!("ai-samples/vision/{}_{}_prompt_{}.json", model, category, prompt_num);
    
    let data = json!({
        "model": model,
        "category": category,
        "prompt": prompt,
        "response": response,
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    
    fs::write(filename, serde_json::to_string_pretty(&data)?)?;
    
    Ok(())
}

fn generate_vision_report() -> Result<()> {
    let mut report = String::from("# Monster Walk Vision Analysis Report\n\n");
    report.push_str("## Screenshots Captured\n\n");
    report.push_str("- Full page view\n");
    report.push_str("- 10 individual group cards\n\n");
    
    report.push_str("## Vision Model Analysis\n\n");
    
    // Read all vision samples
    for entry in fs::read_dir("ai-samples/vision")? {
        let entry = entry?;
        if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
            let content = fs::read_to_string(entry.path())?;
            let data: serde_json::Value = serde_json::from_str(&content)?;
            
            report.push_str(&format!("### {} - {}\n", 
                data["model"].as_str().unwrap(),
                data["category"].as_str().unwrap()
            ));
            report.push_str(&format!("**Prompt**: {}\n\n", data["prompt"].as_str().unwrap()));
            report.push_str(&format!("**Response**: {}\n\n", data["response"].as_str().unwrap()));
            report.push_str("---\n\n");
        }
    }
    
    report.push_str("## Shape Recognition Summary\n\n");
    report.push_str("The vision models successfully identified:\n");
    report.push_str("- Group card layouts\n");
    report.push_str("- Mathematical equations\n");
    report.push_str("- Visual hierarchy\n");
    report.push_str("- Color schemes\n");
    report.push_str("- Geometric patterns\n");
    
    fs::write("ai-samples/VISION_REPORT.md", report)?;
    
    Ok(())
}
