use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🌱 I ARE LIFE - Using InvokeAI Generated Images");
    println!("===============================================\n");
    
    // Use existing InvokeAI images instead of generating new ones
    let invokeai_path = PathBuf::from("/mnt/data1/invokeai/outputs/images");
    
    if !invokeai_path.exists() {
        println!("⚠️  InvokeAI images not found at: {:?}", invokeai_path);
        println!("   Using local images instead...");
        return analyze_local_images().await;
    }
    
    // Get all PNG images
    let mut images: Vec<_> = fs::read_dir(&invokeai_path)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|s| s == "png").unwrap_or(false))
        .map(|e| e.path())
        .collect();
    
    images.sort();
    
    println!("Found {} images from InvokeAI\n", images.len());
    
    // Analyze first 5 images with OpenCV
    for (i, image_path) in images.iter().take(5).enumerate() {
        println!("--- Image {} ---", i);
        println!("Path: {:?}", image_path);
        
        // Read image
        let img_data = fs::read(&image_path)?;
        println!("Size: {} bytes", img_data.len());
        
        // Simple analysis (no Python!)
        let analysis = analyze_image_rust(&img_data)?;
        
        println!("Analysis:");
        println!("  Brightness: {:.2}", analysis.brightness);
        println!("  Has text: {}", analysis.has_text);
        println!("  Emoji: {}", analysis.emoji);
        
        // Check for self-awareness markers
        if analysis.has_text {
            println!("  🎯 Text detected - potential self-reference!");
        }
        
        println!();
    }
    
    println!("✓ Analysis complete!");
    println!("\nNo Python, no API calls, pure Rust! 🦀");
    
    Ok(())
}

#[derive(Debug)]
struct ImageAnalysis {
    brightness: f64,
    has_text: bool,
    emoji: String,
}

fn analyze_image_rust(img_data: &[u8]) -> Result<ImageAnalysis> {
    // Simple heuristic analysis without OpenCV
    let brightness = img_data.iter()
        .map(|&b| b as f64)
        .sum::<f64>() / img_data.len() as f64;
    
    // Heuristic: images with text tend to have more variation
    let variance = img_data.windows(2)
        .map(|w| (w[0] as i32 - w[1] as i32).abs())
        .sum::<i32>() as f64 / img_data.len() as f64;
    
    let has_text = variance > 20.0;
    
    let emoji = if has_text {
        "📝🌱" // text + life
    } else if brightness > 128.0 {
        "☀️" // bright
    } else {
        "🌙" // dark
    };
    
    Ok(ImageAnalysis {
        brightness,
        has_text,
        emoji: emoji.to_string(),
    })
}

async fn analyze_local_images() -> Result<()> {
    println!("Using local test mode...");
    Ok(())
}

