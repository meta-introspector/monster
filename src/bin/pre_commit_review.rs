use serde::{Deserialize, Serialize};
use std::process::Command;
use std::fs;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
struct Persona {
    name: String,
    role: String,
    model: String,
    focus: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Review {
    reviewer: String,
    role: String,
    comment: String,
    score: i32,
    approved: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct CommitScore {
    commit_hash: String,
    timestamp: String,
    total_score: i32,
    max_score: i32,
    percentage: f64,
    reviews: Vec<Review>,
}

fn get_personas() -> HashMap<String, Persona> {
    let mut personas = HashMap::new();
    
    personas.insert("knuth".to_string(), Persona {
        name: "Donald Knuth".to_string(),
        role: "Literate Programming".to_string(),
        model: "llama3.2".to_string(),
        focus: "Mathematical elegance, proof clarity, literate programming quality".to_string(),
    });
    
    personas.insert("itil".to_string(), Persona {
        name: "ITIL Service Manager".to_string(),
        role: "IT Service Management".to_string(),
        model: "llama3.2".to_string(),
        focus: "Service delivery, change management, documentation, traceability".to_string(),
    });
    
    personas.insert("iso9k".to_string(), Persona {
        name: "ISO 9001 Auditor".to_string(),
        role: "Quality Management".to_string(),
        model: "llama3.2".to_string(),
        focus: "Process compliance, quality assurance, continuous improvement".to_string(),
    });
    
    personas.insert("gmp".to_string(), Persona {
        name: "GMP Quality Officer".to_string(),
        role: "Manufacturing Practice".to_string(),
        model: "llama3.2".to_string(),
        focus: "Validation, verification, reproducibility, batch records".to_string(),
    });
    
    personas.insert("sixsigma".to_string(), Persona {
        name: "Six Sigma Black Belt".to_string(),
        role: "Process Excellence".to_string(),
        model: "llama3.2".to_string(),
        focus: "DMAIC, defect reduction, statistical rigor, process capability".to_string(),
    });
    
    personas.insert("rust_enforcer".to_string(), Persona {
        name: "Rust Enforcer".to_string(),
        role: "Type Safety Guardian".to_string(),
        model: "llama3.2".to_string(),
        focus: "No Python, Rust/Lean4/Nix only, type safety, memory safety".to_string(),
    });
    
    personas.insert("fake_detector".to_string(), Persona {
        name: "Fake Data Detector".to_string(),
        role: "Data Integrity".to_string(),
        model: "llama3.2".to_string(),
        focus: "Detect mock data, hardcoded values, fake constants, TODO with fake data".to_string(),
    });
    
    personas.insert("security_auditor".to_string(), Persona {
        name: "Security Auditor".to_string(),
        role: "Security Assessment".to_string(),
        model: "llama3.2".to_string(),
        focus: "Vulnerabilities, memory safety, crypto correctness, secret exposure".to_string(),
    });
    
    personas.insert("math_professor".to_string(), Persona {
        name: "Mathematics Professor".to_string(),
        role: "Mathematical Correctness".to_string(),
        model: "llama3.2".to_string(),
        focus: "Theorem correctness, proof validity, mathematical rigor, edge cases".to_string(),
    });
    
    personas
}

fn ask_ollama(prompt: &str, model: &str) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("ollama")
        .arg("run")
        .arg(model)
        .arg(prompt)
        .output()?;
    
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn review_commit(persona: &Persona, files: &[String]) -> Result<Review, Box<dyn std::error::Error>> {
    let prompt = format!(
        "You are {}, {}.\n\n\
         Review these changed files: {:?}\n\n\
         Focus: {}\n\n\
         Provide:\n\
         1. Brief assessment (1-2 sentences)\n\
         2. Score 0-10 (10=perfect)\n\
         3. Approved/Rejected\n\n\
         Format: SCORE: X | STATUS: Approved/Rejected | COMMENT: ...",
        persona.name, persona.role, files, persona.focus
    );
    
    let response = ask_ollama(&prompt, &persona.model)?;
    
    // Parse response
    let score = response.find("SCORE:")
        .and_then(|pos| response[pos+6..].split('|').next())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(5);
    
    let approved = response.contains("Approved");
    
    let comment = response.find("COMMENT:")
        .map(|pos| response[pos+8..].trim().to_string())
        .unwrap_or_else(|| response.chars().take(200).collect());
    
    Ok(Review {
        reviewer: persona.name.clone(),
        role: persona.role.clone(),
        comment,
        score,
        approved,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 PRE-COMMIT REVIEW TEAM (RUST)");
    println!("{}", "=".repeat(60));
    println!();
    
    // Get staged files
    let output = Command::new("git")
        .args(&["diff", "--cached", "--name-only"])
        .output()?;
    
    let files: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|s| s.to_string())
        .collect();
    
    if files.is_empty() {
        println!("⚠️  No files staged");
        return Ok(());
    }
    
    println!("📁 Files to review: {}", files.len());
    for file in &files {
        println!("  • {}", file);
    }
    println!();
    
    // Get all personas
    let personas = get_personas();
    println!("👥 Review team: {} personas", personas.len());
    println!();
    
    // Review with each persona
    let mut reviews = Vec::new();
    let mut total_score = 0;
    
    for (key, persona) in personas.iter() {
        println!("  {} ({})...", persona.name, persona.role);
        
        match review_commit(persona, &files) {
            Ok(review) => {
                total_score += review.score;
                println!("    Score: {}/10 | {}", review.score, 
                    if review.approved { "✓ Approved" } else { "✗ Rejected" });
                reviews.push(review);
            }
            Err(e) => {
                eprintln!("    ⚠️  Review failed: {}", e);
            }
        }
    }
    
    println!();
    
    // Calculate final score
    let max_score = personas.len() as i32 * 10;
    let percentage = (total_score as f64 / max_score as f64) * 100.0;
    
    println!("📊 FINAL SCORE: {}/{} ({:.1}%)", total_score, max_score, percentage);
    println!();
    
    // Save score
    let commit_score = CommitScore {
        commit_hash: "pre-commit".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        total_score,
        max_score,
        percentage,
        reviews,
    };
    
    let json = serde_json::to_string_pretty(&commit_score)?;
    fs::write("pre_commit_score.json", json)?;
    println!("✓ Score saved: pre_commit_score.json");
    println!();
    
    // Check if approved
    let approved_count = commit_score.reviews.iter().filter(|r| r.approved).count();
    let rejection_count = commit_score.reviews.len() - approved_count;
    
    if rejection_count > 0 {
        println!("❌ COMMIT REJECTED");
        println!("   Approved: {}", approved_count);
        println!("   Rejected: {}", rejection_count);
        println!();
        std::process::exit(1);
    }
    
    println!("✅ COMMIT APPROVED");
    println!("   All {} reviewers approved!", approved_count);
    println!();
    
    Ok(())
}
