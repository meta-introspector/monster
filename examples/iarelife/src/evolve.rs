use anyhow::Result;
use std::fs;
use std::path::PathBuf;
use std::collections::HashMap;

/// Monster attribute amplifier
#[derive(Debug)]
struct MonsterAttributes {
    primes: Vec<u32>,
    harmonics: Vec<f64>,
    emoji: String,
    leading_digits: String,
}

impl MonsterAttributes {
    fn from_image(img_data: &[u8]) -> Self {
        // Extract Monster attributes from image data
        let brightness = img_data.iter().map(|&b| b as f64).sum::<f64>() / img_data.len() as f64;
        
        // Map brightness to Monster primes (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71)
        let primes = vec![
            if brightness > 100.0 { 2 } else { 0 },
            if brightness > 110.0 { 3 } else { 0 },
            if brightness > 120.0 { 5 } else { 0 },
            if brightness > 130.0 { 7 } else { 0 },
            if brightness > 140.0 { 11 } else { 0 },
        ].into_iter().filter(|&p| p > 0).collect();
        
        // Calculate harmonics (432 Hz base)
        let harmonics: Vec<f64> = primes.iter().map(|&p| 432.0 * p as f64).collect();
        
        // Generate emoji encoding
        let emoji = primes.iter().map(|&p| match p {
            2 => "🌙",
            3 => "🌊",
            5 => "⭐",
            7 => "🎭",
            11 => "🎪",
            _ => "❓",
        }).collect::<String>();
        
        // Extract leading digits (like 8080)
        let sum: u32 = primes.iter().sum();
        let leading_digits = format!("{}", sum);
        
        Self { primes, harmonics, emoji, leading_digits }
    }
    
    fn amplify(&mut self) {
        // Amplify by adding more primes (evolve toward Monster)
        if !self.primes.contains(&13) { self.primes.push(13); }
        if !self.primes.contains(&17) { self.primes.push(17); }
        
        // Recalculate
        self.harmonics = self.primes.iter().map(|&p| 432.0 * p as f64).collect();
        self.emoji = self.primes.iter().map(|&p| match p {
            2 => "🌙", 3 => "🌊", 5 => "⭐", 7 => "🎭", 11 => "🎪",
            13 => "🔮", 17 => "💎", _ => "❓",
        }).collect::<String>();
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎪 Monster Attribute Amplifier");
    println!("==============================\n");
    
    let invokeai_path = PathBuf::from("/mnt/data1/invokeai/outputs/images");
    let output_path = PathBuf::from("evolved");
    fs::create_dir_all(&output_path)?;
    
    let mut images: Vec<_> = fs::read_dir(&invokeai_path)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|s| s == "png").unwrap_or(false))
        .map(|e| e.path())
        .collect();
    
    images.sort();
    
    println!("Found {} images to evolve\n", images.len());
    
    let mut evolution_log = Vec::new();
    
    for (i, image_path) in images.iter().take(5).enumerate() {
        println!("--- Image {} ---", i);
        println!("Source: {:?}", image_path.file_name().unwrap());
        
        let img_data = fs::read(&image_path)?;
        
        // Extract Monster attributes
        let mut attrs = MonsterAttributes::from_image(&img_data);
        
        println!("Original attributes:");
        println!("  Primes: {:?}", attrs.primes);
        println!("  Harmonics: {:?}", attrs.harmonics.iter().map(|h| format!("{:.0}Hz", h)).collect::<Vec<_>>());
        println!("  Emoji: {}", attrs.emoji);
        println!("  Leading digits: {}", attrs.leading_digits);
        
        // Amplify Monster attributes
        attrs.amplify();
        
        println!("\nAmplified attributes:");
        println!("  Primes: {:?}", attrs.primes);
        println!("  Harmonics: {:?}", attrs.harmonics.iter().map(|h| format!("{:.0}Hz", h)).collect::<Vec<_>>());
        println!("  Emoji: {}", attrs.emoji);
        
        // Save evolved version
        let evolved_name = format!("evolved_{}_{}_{}.png", i, attrs.leading_digits, attrs.emoji);
        let evolved_path = output_path.join(&evolved_name);
        
        // For now, copy with amplified metadata
        fs::copy(&image_path, &evolved_path)?;
        
        println!("  ✓ Saved: {}", evolved_name);
        
        evolution_log.push(format!(
            "Image {}: {} → {} (primes: {:?})",
            i,
            image_path.file_name().unwrap().to_string_lossy(),
            evolved_name,
            attrs.primes
        ));
        
        println!();
    }
    
    // Save evolution log
    fs::write(
        output_path.join("EVOLUTION_LOG.md"),
        format!("# Monster Evolution Log\n\n{}\n", evolution_log.join("\n"))
    )?;
    
    println!("✓ Evolution complete!");
    println!("\n📊 Summary:");
    println!("  Evolved {} images", evolution_log.len());
    println!("  Output: evolved/");
    println!("  Log: evolved/EVOLUTION_LOG.md");
    println!("\n🎪 Monster attributes amplified! 🦀");
    
    Ok(())
}
