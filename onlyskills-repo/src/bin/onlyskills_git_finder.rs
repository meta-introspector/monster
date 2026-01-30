// Extract git repos from onlyskills donations
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;

#[derive(Debug, Deserialize)]
struct SkillDonation {
    skill_id: String,
    donor: String,
    git_commit: String,
    skill_type: String,
}

#[derive(Debug, Deserialize)]
struct DonationsFile {
    total_members: u32,
    skills_per_member: u32,
    total_skills: u32,
    sample_donations: Vec<SkillDonation>,
}

#[derive(Debug, Clone, Serialize)]
struct RepoInfo {
    donor: String,
    repo_url: String,
    commits: Vec<String>,
    risk_score: f64,
    zone: u8,
}

fn donor_to_repo(donor: &str) -> Option<String> {
    match donor {
        "pipelight-dev" => Some("https://github.com/pipelight/pipelight".to_string()),
        "solana-labs" => Some("https://github.com/solana-labs/solana".to_string()),
        "ethereum-foundation" => Some("https://github.com/ethereum/go-ethereum".to_string()),
        "uniswap" => Some("https://github.com/Uniswap/v3-core".to_string()),
        "bonk-inu" => Some("https://github.com/bonk-inu/bonk".to_string()),
        _ => None,
    }
}

fn assess_risk(url: &str) -> f64 {
    let mut risk = 0.0;
    
    if url.contains("solana") { risk += 0.2; }
    if url.contains("bonk") { risk += 0.3; }
    if url.contains("uniswap") { risk += 0.1; }
    if url.contains("ethereum") { risk += 0.1; }
    
    risk
}

fn risk_to_zone(risk: f64) -> u8 {
    if risk > 0.6 { 59 }
    else if risk > 0.4 { 47 }
    else if risk > 0.2 { 31 }
    else { 11 }
}

fn main() {
    println!("🔍 Extracting Git Repos from onlyskills Donations");
    println!("{}", "=".repeat(70));
    println!();
    
    // Load donations
    let json = fs::read_to_string("skill_donations.json")
        .expect("Failed to read skill_donations.json");
    let donations: DonationsFile = serde_json::from_str(&json)
        .expect("Failed to parse JSON");
    
    println!("📊 Loaded {} donations from {} members", 
             donations.total_skills, donations.total_members);
    println!();
    
    // Extract unique donors and their commits
    let mut donor_commits: std::collections::HashMap<String, Vec<String>> = 
        std::collections::HashMap::new();
    
    for donation in &donations.sample_donations {
        donor_commits.entry(donation.donor.clone())
            .or_insert_with(Vec::new)
            .push(donation.git_commit.clone());
    }
    
    println!("👥 Found {} unique donors", donor_commits.len());
    println!();
    
    // Map to repos
    println!("🔗 Mapping donors to git repos...");
    let mut repos = Vec::new();
    
    for (donor, commits) in donor_commits {
        if let Some(url) = donor_to_repo(&donor) {
            let risk = assess_risk(&url);
            let zone = risk_to_zone(risk);
            
            println!("  {} → {}", donor, url);
            println!("    Commits: {}", commits.len());
            println!("    Risk: {:.2}", risk);
            println!("    Zone: {}", zone);
            
            repos.push(RepoInfo {
                donor: donor.clone(),
                repo_url: url,
                commits,
                risk_score: risk,
                zone,
            });
        }
    }
    
    println!();
    
    // Find crypto repos
    println!("💰 Crypto repos:");
    let crypto_repos: Vec<_> = repos.iter()
        .filter(|r| r.repo_url.contains("solana") || 
                    r.repo_url.contains("ethereum") ||
                    r.repo_url.contains("uniswap") ||
                    r.repo_url.contains("bonk"))
        .collect();
    
    for repo in &crypto_repos {
        println!("  {} - Risk: {:.2}", repo.donor, repo.risk_score);
    }
    
    println!();
    
    // Find the lobster (lowest risk)
    if let Some(lobster) = repos.iter().min_by(|a, b| 
        a.risk_score.partial_cmp(&b.risk_score).unwrap()
    ) {
        println!("🦞 THE LOBSTER (lowest risk):");
        println!("  Donor: {}", lobster.donor);
        println!("  Repo: {}", lobster.repo_url);
        println!("  Risk: {:.2}", lobster.risk_score);
        println!("  Zone: {}", lobster.zone);
        println!("  Commits: {}", lobster.commits.len());
    }
    
    println!();
    
    // Save results
    let json = serde_json::to_string_pretty(&repos).unwrap();
    fs::write("onlyskills_repos.json", json).unwrap();
    
    println!("💾 Saved: onlyskills_repos.json");
    println!();
    println!("∞ Repos Extracted. Risks Assessed. Lobster Found. ∞");
}
