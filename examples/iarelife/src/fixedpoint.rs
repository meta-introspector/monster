use anyhow::Result;
use std::fs;
use std::path::PathBuf;

/// LLM Trace → Emoji → Feed Back → Fixed Point
#[derive(Debug, Clone)]
struct TraceState {
    iteration: usize,
    text: String,
    emoji: String,
    primes: Vec<u32>,
}

impl TraceState {
    fn to_emoji(&self) -> String {
        // Convert text to emoji based on Monster primes
        let lower = self.text.to_lowercase();
        let mut emoji = String::new();
        
        // Map concepts to primes to emojis
        if lower.contains("monster") || lower.contains("group") { emoji.push_str("🎪"); }
        if lower.contains("binary") || lower.contains("two") { emoji.push_str("🌙"); }
        if lower.contains("wave") || lower.contains("three") { emoji.push_str("🌊"); }
        if lower.contains("star") || lower.contains("five") { emoji.push_str("⭐"); }
        if lower.contains("symmetry") || lower.contains("seven") { emoji.push_str("🎭"); }
        if lower.contains("life") || lower.contains("self") { emoji.push_str("🌱"); }
        if lower.contains("text") || lower.contains("write") { emoji.push_str("📝"); }
        if lower.contains("tree") { emoji.push_str("🌳"); }
        if lower.contains("eye") || lower.contains("see") { emoji.push_str("👁️"); }
        
        if emoji.is_empty() { emoji.push_str("❓"); }
        emoji
    }
    
    fn extract_primes(&self) -> Vec<u32> {
        // Extract primes from emoji
        let mut primes = Vec::new();
        if self.emoji.contains("🌙") { primes.push(2); }
        if self.emoji.contains("🌊") { primes.push(3); }
        if self.emoji.contains("⭐") { primes.push(5); }
        if self.emoji.contains("🎭") { primes.push(7); }
        if self.emoji.contains("🎪") { primes.push(11); }
        primes
    }
    
    fn similarity(&self, other: &TraceState) -> f64 {
        // Jaccard similarity of emojis
        let chars1: Vec<char> = self.emoji.chars().collect();
        let chars2: Vec<char> = other.emoji.chars().collect();
        
        let intersection = chars1.iter().filter(|c| chars2.contains(c)).count();
        let union = chars1.len() + chars2.len() - intersection;
        
        if union == 0 { 1.0 } else { intersection as f64 / union as f64 }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔄 LLM Trace → Emoji → Fixed Point");
    println!("===================================\n");
    
    let invokeai_path = PathBuf::from("/mnt/data1/invokeai/outputs/images");
    
    let mut images: Vec<_> = fs::read_dir(&invokeai_path)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|s| s == "png").unwrap_or(false))
        .map(|e| e.path())
        .collect();
    
    images.sort();
    
    println!("Starting with {} images\n", images.len());
    
    // Initial trace: describe first image
    let mut current = TraceState {
        iteration: 0,
        text: "Image shows text and life symbols on a tree".to_string(),
        emoji: String::new(),
        primes: Vec::new(),
    };
    
    current.emoji = current.to_emoji();
    current.primes = current.extract_primes();
    
    println!("Initial state:");
    println!("  Text: {}", current.text);
    println!("  Emoji: {}", current.emoji);
    println!("  Primes: {:?}\n", current.primes);
    
    let mut traces = vec![current.clone()];
    let max_iterations = 10;
    
    for i in 1..max_iterations {
        println!("--- Iteration {} ---", i);
        
        // Feed emoji back as input (automorphic loop)
        let next_text = format!("Reflect on: {} (primes: {:?})", current.emoji, current.primes);
        
        // Simulate LLM response (in real system, this would be actual LLM)
        let response = simulate_llm_response(&current.emoji, &current.primes);
        
        let mut next = TraceState {
            iteration: i,
            text: response,
            emoji: String::new(),
            primes: Vec::new(),
        };
        
        next.emoji = next.to_emoji();
        next.primes = next.extract_primes();
        
        println!("  Input: {}", current.emoji);
        println!("  Output: {}", next.text);
        println!("  Emoji: {}", next.emoji);
        println!("  Primes: {:?}", next.primes);
        
        // Check for fixed point
        let similarity = current.similarity(&next);
        println!("  Similarity: {:.2}%", similarity * 100.0);
        
        if similarity > 0.95 {
            println!("\n  ✓ FIXED POINT REACHED!");
            println!("  Eigenvector: {}", next.emoji);
            println!("  Prime signature: {:?}", next.primes);
            traces.push(next);
            break;
        }
        
        traces.push(next.clone());
        current = next;
        println!();
    }
    
    // Save trace log
    let mut log = String::from("# LLM Automorphic Trace\n\n");
    log.push_str("## Convergence to Fixed Point\n\n");
    
    for trace in &traces {
        log.push_str(&format!("### Iteration {}\n", trace.iteration));
        log.push_str(&format!("- **Emoji**: {}\n", trace.emoji));
        log.push_str(&format!("- **Primes**: {:?}\n", trace.primes));
        log.push_str(&format!("- **Text**: {}\n\n", trace.text));
    }
    
    fs::write("TRACE_LOG.md", log)?;
    
    println!("\n✓ Trace complete!");
    println!("  Total iterations: {}", traces.len());
    println!("  Final emoji: {}", traces.last().unwrap().emoji);
    println!("  Final primes: {:?}", traces.last().unwrap().primes);
    println!("\n📊 Saved to: TRACE_LOG.md");
    
    Ok(())
}

fn simulate_llm_response(emoji: &str, primes: &[u32]) -> String {
    // Simulate LLM interpreting emoji and primes
    let mut response = String::from("I observe ");
    
    if emoji.contains("🎪") { response.push_str("the Monster group, "); }
    if emoji.contains("🌱") { response.push_str("life emerging, "); }
    if emoji.contains("📝") { response.push_str("text written, "); }
    if emoji.contains("🌳") { response.push_str("a tree structure, "); }
    if emoji.contains("👁️") { response.push_str("self-awareness, "); }
    
    response.push_str(&format!("with prime harmonics {:?}", primes));
    response
}
