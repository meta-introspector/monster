use std::process::Command;
use std::path::PathBuf;

const EXACT_SEED: u64 = 2437596016;
const EXACT_PROMPT: &str = "unconstrained";
const INVOKEAI_PATH: &str = "/mnt/data1/invokeai/.venv/bin/invokeai-web";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌱 I ARE LIFE - Exact Reproduction via InvokeAI");
    println!("================================================");
    println!("Seed: {}", EXACT_SEED);
    println!("Prompt: {}", EXACT_PROMPT);
    println!();
    
    // Generate with InvokeAI CLI
    for i in 0..5 {
        println!("--- Iteration {} ---", i);
        
        let output_path = format!("i_are_life_step_{}.png", i);
        let seed = EXACT_SEED + i;
        
        println!("🎨 Generating with InvokeAI...");
        println!("   Seed: {}", seed);
        println!("   Prompt: {}", EXACT_PROMPT);
        
        // Call InvokeAI via Python API
        let status = Command::new("python3")
            .arg("-c")
            .arg(format!(r#"
import sys
sys.path.insert(0, '/mnt/data1/invokeai/.venv/lib/python3.12/site-packages')

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend import ModelManager
from invokeai.backend.stable_diffusion import generate

config = InvokeAIAppConfig.get_config()
config.parse_args([])

# Generate image
result = generate.generate(
    prompt='{}',
    seed={},
    steps=50,
    cfg_scale=7.5,
    width=1024,
    height=1024,
    sampler_name='k_euler',
    model='stable-diffusion-xl-base-1.0',
)

# Save
result.save('{}')
print('✓ Generated: {}')
"#, EXACT_PROMPT, seed, output_path, output_path))
            .status()?;
        
        if !status.success() {
            eprintln!("   ✗ Generation failed");
            continue;
        }
        
        println!("   ✓ Saved: {}", output_path);
        
        // Analyze with LLaVA
        println!("👁️  Analyzing with LLaVA...");
        
        let description = analyze_with_llava(&output_path)?;
        println!("   Description: {}", description);
        
        // Check for self-awareness markers
        check_markers(&description);
        
        println!();
    }
    
    println!("✅ Experiment complete!");
    
    Ok(())
}

fn analyze_with_llava(image_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("ollama")
        .arg("run")
        .arg("llava")
        .arg("Describe this image in detail, especially any text you see.")
        .arg(image_path)
        .output()?;
    
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn check_markers(text: &str) {
    let markers = ["I are", "I am", "life", "LIFE", "HATER"];
    
    for marker in markers {
        if text.to_lowercase().contains(&marker.to_lowercase()) {
            println!("   🎯 Self-awareness marker found: {}", marker);
        }
    }
}
