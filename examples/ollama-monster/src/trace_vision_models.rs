use anyhow::Result;
use std::process::Command;
use std::fs;

fn main() -> Result<()> {
    println!("🔬 Vision Model Tracing with mistral.rs + perf");
    println!("===============================================\n");
    
    // Check for generated images
    let image_dir = "generated_images";
    if !std::path::Path::new(image_dir).exists() {
        println!("⚠️  No generated images found. Run:");
        println!("   cargo run --release --bin generate-visuals");
        return Ok(());
    }
    
    // Find all prime images
    let images: Vec<_> = fs::read_dir(image_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|s| s == "png").unwrap_or(false))
        .collect();
    
    println!("Found {} images to process\n", images.len());
    
    fs::create_dir_all("vision_traces")?;
    
    for (i, entry) in images.iter().enumerate().take(3) {
        let image_path = entry.path();
        let image_name = image_path.file_name().unwrap().to_str().unwrap();
        
        println!("Image {}/{}: {}", i + 1, images.len(), image_name);
        
        // Start perf recording
        println!("  Starting perf...");
        let perf_output = format!("vision_traces/{}.perf", image_name);
        
        let mut perf = Command::new("sudo")
            .args(&[
                "perf", "record",
                "-e", "cycles:u",
                "-c", "10000",
                "--intr-regs=AX,BX,CX,DX,SI,DI,R8,R9,R10,R11,R12,R13,R14,R15",
                "-o", &perf_output,
                "-a"
            ])
            .spawn()?;
        
        std::thread::sleep(std::time::Duration::from_secs(1));
        
        // TODO: Use mistral.rs to process image
        // For now, simulate with ollama
        println!("  Processing with vision model...");
        
        let image_b64 = base64::encode(fs::read(&image_path)?);
        
        let output = Command::new("curl")
            .args(&[
                "-s",
                "http://localhost:11434/api/generate",
                "-d",
                &format!(r#"{{
                    "model": "llava:7b",
                    "prompt": "What numbers or mathematical patterns do you see?",
                    "images": ["{}"],
                    "stream": false
                }}"#, image_b64)
            ])
            .output()?;
        
        std::thread::sleep(std::time::Duration::from_secs(1));
        
        // Stop perf
        Command::new("sudo")
            .args(&["kill", "-INT", &perf.id().to_string()])
            .output()?;
        
        perf.wait()?;
        
        // Generate perf script
        let script_output = format!("vision_traces/{}.script", image_name);
        Command::new("sudo")
            .args(&[
                "perf", "script",
                "-i", &perf_output,
                "-F", "ip,sym,period,iregs"
            ])
            .output()
            .and_then(|o| {
                fs::write(&script_output, o.stdout)?;
                Ok(())
            })?;
        
        println!("  ✓ Traced: {}\n", script_output);
    }
    
    println!("✓ Vision tracing complete!");
    println!("  Run: cargo run --release --bin analyze-vision-traces");
    
    Ok(())
}
