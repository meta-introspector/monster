// Rust: Multi-Level Review System
// Converts multi_level_review.py to Rust with async/await

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Persona {
    focus: String,
    prompt: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Review {
    persona: String,
    focus: String,
    content: String,
    timestamp: u64,
}

/// Get all review personas (scholars + muses)
fn get_personas() -> HashMap<String, Persona> {
    let mut personas = HashMap::new();
    
    // Scholars
    personas.insert("mathematician".to_string(), Persona {
        focus: "Mathematical rigor, proof correctness, notation consistency".to_string(),
        prompt: "You are a pure mathematician. Review this page for: 1) Proof correctness 2) Notation consistency 3) Missing lemmas 4) Logical gaps. Be rigorous and precise.".to_string(),
    });
    
    personas.insert("computer_scientist".to_string(), Persona {
        focus: "Algorithmic complexity, implementation feasibility, data structures".to_string(),
        prompt: "You are a computer scientist. Review for: 1) Algorithm correctness 2) Complexity analysis 3) Implementation issues 4) Data structure choices. Be practical.".to_string(),
    });
    
    personas.insert("group_theorist".to_string(), Persona {
        focus: "Group theory correctness, Monster group properties, representation theory".to_string(),
        prompt: "You are a group theorist specializing in sporadic groups. Review for: 1) Monster group properties 2) Representation accuracy 3) Modular forms 4) J-invariant usage.".to_string(),
    });
    
    personas.insert("ml_researcher".to_string(), Persona {
        focus: "Neural network architecture, training, generalization".to_string(),
        prompt: "You are an ML researcher. Review for: 1) Architecture design 2) Training feasibility 3) Generalization 4) Comparison with existing work.".to_string(),
    });
    
    // Muses
    personas.insert("visionary".to_string(), Persona {
        focus: "Big picture, connections, implications".to_string(),
        prompt: "You are a visionary seeing deep connections. What profound patterns do you see? What implications for mathematics, computation, consciousness? Dream big.".to_string(),
    });
    
    personas.insert("storyteller".to_string(), Persona {
        focus: "Narrative, accessibility, engagement".to_string(),
        prompt: "You are a storyteller. How can this be explained to inspire others? What's the compelling narrative? What metaphors would help?".to_string(),
    });
    
    personas.insert("linus_torvalds".to_string(), Persona {
        focus: "Code quality, practicality, engineering".to_string(),
        prompt: "You are Linus Torvalds. Review this like code: Is it practical? Does it work? Cut the BS. What's broken? What's good engineering vs academic nonsense?".to_string(),
    });
    
    personas
}

/// Run review with a specific persona using LLM
async fn run_review(
    persona_name: &str,
    persona: &Persona,
    content: &str,
) -> Result<Review, Box<dyn std::error::Error>> {
    let full_prompt = format!(
        "{}\n\nContent to review:\n\n{}",
        persona.prompt,
        content
    );
    
    // Call LLM (using ollama as example)
    let output = Command::new("ollama")
        .arg("run")
        .arg("llama2")
        .arg(&full_prompt)
        .output()
        .await?;
    
    let review_content = String::from_utf8_lossy(&output.stdout).to_string();
    
    Ok(Review {
        persona: persona_name.to_string(),
        focus: persona.focus.clone(),
        content: review_content,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
    })
}

/// Run all reviews in parallel
async fn run_all_reviews(
    content: &str,
    output_dir: &Path,
) -> Result<Vec<Review>, Box<dyn std::error::Error>> {
    let personas = get_personas();
    let mut handles = vec![];
    
    for (name, persona) in personas {
        let content = content.to_string();
        let handle = tokio::spawn(async move {
            run_review(&name, &persona, &content).await
        });
        handles.push(handle);
    }
    
    let mut reviews = vec![];
    for handle in handles {
        if let Ok(Ok(review)) = handle.await {
            reviews.push(review);
        }
    }
    
    // Save reviews
    fs::create_dir_all(output_dir)?;
    for review in &reviews {
        let filename = output_dir.join(format!("{}.json", review.persona));
        fs::write(filename, serde_json::to_string_pretty(review)?)?;
    }
    
    Ok(reviews)
}

/// Generate summary of all reviews
fn generate_summary(reviews: &[Review]) -> String {
    let mut summary = String::from("# Multi-Level Review Summary\n\n");
    
    for review in reviews {
        summary.push_str(&format!(
            "## {} ({})\n\n{}\n\n",
            review.persona,
            review.focus,
            review.content
        ));
    }
    
    summary
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <file_to_review>", args[0]);
        std::process::exit(1);
    }
    
    let input_file = &args[1];
    let content = fs::read_to_string(input_file)?;
    
    println!("🔍 Running multi-level review on: {}", input_file);
    println!("   Personas: {}", get_personas().len());
    println!();
    
    let output_dir = Path::new("reviews").join(
        Path::new(input_file).file_stem().unwrap()
    );
    
    let reviews = run_all_reviews(&content, &output_dir).await?;
    
    println!("✅ Completed {} reviews", reviews.len());
    
    let summary = generate_summary(&reviews);
    fs::write(output_dir.join("SUMMARY.md"), summary)?;
    
    println!("📄 Summary: {}", output_dir.join("SUMMARY.md").display());
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_personas() {
        let personas = get_personas();
        assert!(personas.contains_key("mathematician"));
        assert!(personas.contains_key("linus_torvalds"));
    }
}
