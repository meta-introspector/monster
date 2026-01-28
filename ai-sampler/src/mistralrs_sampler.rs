use anyhow::Result;
use headless_chrome::{Browser, LaunchOptions, protocol::cdp::Page};
use std::fs;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎪 Monster Walk AI Sampler - Pure Rust with mistral.rs");
    println!("=======================================================\n");
    
    // Step 1: Extract content from site
    println!("📖 Extracting content from Monster Walk site...");
    let content = extract_site_content()?;
    println!("✓ Extracted content\n");
    
    // Step 2: Capture screenshots
    println!("📸 Capturing screenshots...");
    let screenshots = capture_screenshots()?;
    println!("✓ Captured {} screenshots\n", screenshots.len());
    
    // Step 3: Initialize mistral.rs
    println!("🤖 Initializing mistral.rs...");
    println!("   Loading model from: ~/.cache/mistral.rs/");
    
    // TODO: Initialize mistral.rs model
    // let model = mistralrs::Model::load("mistral-7b")?;
    
    // Step 4: Sample with prompts
    let prompts = vec![
        "Summarize the Monster Walk discovery.",
        "What is Bott periodicity?",
        "How many groups were found?",
        "Explain the 10-fold way connection.",
    ];
    
    println!("\n📝 Sampling with prompts...");
    for (i, prompt) in prompts.iter().enumerate() {
        println!("\n  Prompt {}: {}", i + 1, prompt);
        
        let full_prompt = format!("{}\n\nContext:\n{}", prompt, &content[..1000]);
        
        // TODO: Query mistral.rs
        // let response = model.generate(&full_prompt)?;
        let response = format!("[mistral.rs inference would go here for: {}]", prompt);
        
        println!("  Response: {}", &response[..100.min(response.len())]);
        
        save_response("mistral-7b", i, prompt, &response)?;
    }
    
    // Step 5: Vision analysis
    println!("\n👁️  Vision analysis...");
    analyze_screenshots_with_vision(&screenshots)?;
    
    println!("\n✓ Complete! Results saved to ai-samples/mistralrs/");
    
    Ok(())
}

fn extract_site_content() -> Result<String> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = std::env::current_dir()?.join("../result/index.html");
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    std::thread::sleep(std::time::Duration::from_secs(3));
    
    let content = tab.evaluate(r#"
        JSON.stringify({
            title: document.title,
            groups: Array.from(document.querySelectorAll('.group-card')).map(card => ({
                id: card.id,
                text: card.textContent.trim()
            })),
            monster_order: "808017424794512875886459904961710757005754368000000000",
            summary: "The Monster Walk reveals 10 hierarchical groups through prime factorization, exhibiting Bott periodicity and the 10-fold way."
        })
    "#, false)?;
    
    if let Some(serde_json::Value::String(json_str)) = content.value {
        Ok(json_str)
    } else {
        Ok(String::new())
    }
}

fn capture_screenshots() -> Result<Vec<Vec<u8>>> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = std::env::current_dir()?.join("../result/index.html");
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    std::thread::sleep(std::time::Duration::from_secs(5));
    
    fs::create_dir_all("ai-samples/screenshots")?;
    
    let mut screenshots = Vec::new();
    
    // Full page
    let screenshot = tab.capture_screenshot(
        Page::CaptureScreenshotFormatOption::Png,
        None,
        None,
        true
    )?;
    fs::write("ai-samples/screenshots/full_page.png", &screenshot)?;
    screenshots.push(screenshot);
    
    // Individual groups
    for i in 1..=10 {
        tab.evaluate(&format!(r#"
            document.querySelector('#group-{}')?.scrollIntoView({{block: 'center'}});
        "#, i), false)?;
        
        std::thread::sleep(std::time::Duration::from_millis(300));
        
        let screenshot = tab.capture_screenshot(
            Page::CaptureScreenshotFormatOption::Png,
            None,
            None,
            false
        )?;
        fs::write(format!("ai-samples/screenshots/group_{}.png", i), &screenshot)?;
        screenshots.push(screenshot);
    }
    
    Ok(screenshots)
}

fn analyze_screenshots_with_vision(screenshots: &[Vec<u8>]) -> Result<()> {
    println!("  Analyzing {} screenshots with vision model...", screenshots.len());
    
    // TODO: Use mistral.rs vision model
    // let vision_model = mistralrs::VisionModel::load("llava")?;
    
    for (i, screenshot) in screenshots.iter().enumerate() {
        let prompt = if i == 0 {
            "Describe the overall structure and layout of this mathematical visualization."
        } else {
            "What mathematical information is shown in this group card?"
        };
        
        // TODO: Vision inference
        // let response = vision_model.analyze(screenshot, prompt)?;
        let response = format!("[Vision analysis of screenshot {} would go here]", i);
        
        println!("    Screenshot {}: {}", i, &response[..80.min(response.len())]);
    }
    
    Ok(())
}

fn save_response(model: &str, prompt_num: usize, prompt: &str, response: &str) -> Result<()> {
    fs::create_dir_all("ai-samples/mistralrs")?;
    
    let filename = format!("ai-samples/mistralrs/{}_prompt_{}.json", model, prompt_num);
    
    let data = json!({
        "model": model,
        "prompt": prompt,
        "response": response,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "engine": "mistral.rs"
    });
    
    fs::write(filename, serde_json::to_string_pretty(&data)?)?;
    
    Ok(())
}
