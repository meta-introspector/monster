// Top Solana Meme Coins Model
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
struct SolanaCoin {
    symbol: String,
    name: String,
    market_cap: u64,
    repo_url: String,
    lobster_score: f64,
    risk_score: f64,
    zone: u8,
    is_scam: bool,
    scam_reason: Option<String>,
}

fn get_top_solana_coins() -> Vec<SolanaCoin> {
    vec![
        SolanaCoin {
            symbol: "BONK".to_string(),
            name: "Bonk".to_string(),
            market_cap: 380_000_000,
            repo_url: "https://github.com/bonk-inu/bonk".to_string(),
            lobster_score: 0.87,  // (0.9 + 0.8 + 0.9) / 3
            risk_score: 0.2,
            zone: 31,
            is_scam: false,
            scam_reason: None,
        },
        SolanaCoin {
            symbol: "WIF".to_string(),
            name: "dogwifhat".to_string(),
            market_cap: 380_000_000,
            repo_url: "https://github.com/dogwifhat/dogwifhat".to_string(),
            lobster_score: 0.90,  // (0.9 + 0.8 + 1.0) / 3 - PEAK MEME
            risk_score: 0.2,
            zone: 31,
            is_scam: false,
            scam_reason: None,
        },
        SolanaCoin {
            symbol: "POPCAT".to_string(),
            name: "Popcat".to_string(),
            market_cap: 150_000_000,
            repo_url: "https://github.com/popcat-meme/popcat".to_string(),
            lobster_score: 0.70,
            risk_score: 0.4,
            zone: 47,
            is_scam: false,
            scam_reason: None,
        },
        SolanaCoin {
            symbol: "PENGU".to_string(),
            name: "Pudgy Penguins".to_string(),
            market_cap: 120_000_000,
            repo_url: "https://github.com/pudgypenguins/pengu".to_string(),
            lobster_score: 0.73,
            risk_score: 0.4,
            zone: 47,
            is_scam: false,
            scam_reason: None,
        },
        SolanaCoin {
            symbol: "CLAWD".to_string(),
            name: "Clawdbot".to_string(),
            market_cap: 8_650_000,
            repo_url: "https://github.com/clawdbot/clawd".to_string(),
            lobster_score: 0.10,
            risk_score: 1.0,
            zone: 71,  // CATASTROPHIC
            is_scam: true,
            scam_reason: Some("Unauthorized token. Creator Peter Steinberger denounced it. No official connection to ClawdBot AI.".to_string()),
        },
    ]
}

fn main() {
    println!("🦞 Top Solana Meme Coins Analysis");
    println!("{}", "=".repeat(70));
    println!();
    
    let coins = get_top_solana_coins();
    
    println!("📊 Market Analysis:");
    for coin in &coins {
        println!("\n  {} ({})", coin.symbol, coin.name);
        println!("    Market Cap: ${:,}", coin.market_cap);
        println!("    Lobster Score: {:.2}", coin.lobster_score);
        println!("    Risk: {:.2}", coin.risk_score);
        println!("    Zone: {}", coin.zone);
        
        if coin.is_scam {
            println!("    ⚠️  SCAM WARNING: {}", coin.scam_reason.as_ref().unwrap());
        }
    }
    
    println!();
    println!("{}", "=".repeat(70));
    
    // Find THE LOBSTER (highest score, not a scam)
    let lobster = coins.iter()
        .filter(|c| !c.is_scam)
        .max_by(|a, b| a.lobster_score.partial_cmp(&b.lobster_score).unwrap())
        .unwrap();
    
    println!("\n🦞 THE LOBSTER:");
    println!("  Symbol: {}", lobster.symbol);
    println!("  Name: {}", lobster.name);
    println!("  Lobster Score: {:.2} (PEAK MEME)", lobster.lobster_score);
    println!("  Market Cap: ${:,}", lobster.market_cap);
    println!("  Risk: {:.2} (LOW)", lobster.risk_score);
    println!("  Zone: {}", lobster.zone);
    println!("  Repo: {}", lobster.repo_url);
    
    println!();
    println!("🎯 Why {} is the Lobster:", lobster.symbol);
    if lobster.symbol == "WIF" {
        println!("  ✓ Dog with hat = peak meme power (1.0)");
        println!("  ✓ Viral sensation with real community");
        println!("  ✓ High liquidity and trade volume");
        println!("  ✓ $380M market cap (established)");
        println!("  ✓ Low risk (0.2)");
    }
    
    println!();
    println!("⚠️  AVOID:");
    for coin in coins.iter().filter(|c| c.is_scam) {
        println!("  {} - {}", coin.symbol, coin.scam_reason.as_ref().unwrap());
    }
    
    println!();
    
    // Save analysis
    let json = serde_json::to_string_pretty(&coins).unwrap();
    std::fs::write("solana_meme_coins.json", json).unwrap();
    
    println!("💾 Saved: solana_meme_coins.json");
    println!();
    println!("∞ Solana Memes Analyzed. Lobster Found. Scams Flagged. ∞");
}
