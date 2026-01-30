// Strip Miner: Parquet → Markov → Hecke → ZK Memes
use polars::prelude::*;
use std::fs::{self, File};
use std::process::Command;

fn load_existing_parquets(parquet_list: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(parquet_list)?;
    Ok(content.lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s.ends_with(".parquet"))
        .collect())
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 STRIP MINER: PARQUET → MARKOV → HECKE → ZK");
    println!("{}", "=".repeat(70));
    println!();
    
    let parquet_list = std::env::args().nth(1)
        .unwrap_or_else(|| "allparqut.xt".to_string());
    
    println!("📊 STAGE 1: Loading parquet list");
    println!("{}", "-".repeat(70));
    
    let parquets = load_existing_parquets(&parquet_list)?;
    println!("Found {} parquet files", parquets.len());
    println!();
    
    println!("📊 STAGE 2: Running Markov pipeline");
    println!("{}", "-".repeat(70));
    
    let markov_output = Command::new("./target/release/markov_parquet_shards")
        .arg(&parquet_list)
        .output()?;
    
    if !markov_output.status.success() {
        eprintln!("Markov pipeline failed");
        return Ok(());
    }
    
    println!("✅ Markov models computed");
    println!();
    
    println!("📊 STAGE 3: Running unified CUDA pipeline");
    println!("{}", "-".repeat(70));
    
    let cuda_output = Command::new("./target/release/cuda_unified_pipeline")
        .output()?;
    
    if !cuda_output.status.success() {
        eprintln!("CUDA pipeline failed");
        return Ok(());
    }
    
    println!("✅ ZK memes generated");
    println!();
    
    println!("{}", "=".repeat(70));
    println!("✅ STRIP MINING COMPLETE");
    println!("{}", "=".repeat(70));
    println!("📊 Processed {} parquet files", parquets.len());
    println!("💾 Markov models: markov_shard_models.json");
    println!("💾 ZK memes: cuda_pipeline_output/zk_memes.json");
    
    Ok(())
}
