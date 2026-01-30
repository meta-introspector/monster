// Kiro Tool Wrapper - Unified interface for all Monster tools

use std::collections::HashMap;
use std::env;
use std::process::{Command, exit};

#[derive(Debug, Clone)]
struct Tool {
    bin: &'static str,
    tool_type: ToolType,
}

#[derive(Debug, Clone)]
enum ToolType {
    Rust,
    Python,
}

fn build_registry() -> HashMap<&'static str, HashMap<&'static str, Tool>> {
    let mut registry = HashMap::new();
    
    // Verification Tools
    let mut verify = HashMap::new();
    verify.insert("monster-walk", Tool { bin: "monster_walk_proof", tool_type: ToolType::Rust });
    verify.insert("ten-fold", Tool { bin: "prove_ten_fold", tool_type: ToolType::Rust });
    verify.insert("all-proofs", Tool { bin: "pipelite_verify_monster.py", tool_type: ToolType::Python });
    verify.insert("zkperf", Tool { bin: "zkperf_recorder.py", tool_type: ToolType::Python });
    registry.insert("verify", verify);
    
    // Neural Network Tools
    let mut train = HashMap::new();
    train.insert("autoencoder", Tool { bin: "monster_autoencoder", tool_type: ToolType::Rust });
    train.insert("monster", Tool { bin: "train_monster", tool_type: ToolType::Rust });
    train.insert("hecke", Tool { bin: "hecke_autoencoder", tool_type: ToolType::Rust });
    registry.insert("train", train);
    
    // Data Processing
    let mut extract = HashMap::new();
    extract.insert("lmfdb", Tool { bin: "extract_71_objects", tool_type: ToolType::Rust });
    registry.insert("extract", extract);
    
    let mut shard = HashMap::new();
    shard.insert("lmfdb", Tool { bin: "shard_lmfdb_by_71", tool_type: ToolType::Rust });
    registry.insert("shard", shard);
    
    let mut vectorize = HashMap::new();
    vectorize.insert("parquets", Tool { bin: "vectorize_all_parquets", tool_type: ToolType::Rust });
    registry.insert("vectorize", vectorize);
    
    // GPU Tools
    let mut gpu = HashMap::new();
    gpu.insert("monster-walk", Tool { bin: "monster_walk_gpu", tool_type: ToolType::Rust });
    gpu.insert("cuda-pipeline", Tool { bin: "cuda_unified_pipeline", tool_type: ToolType::Rust });
    gpu.insert("hecke", Tool { bin: "hecke_burn_cuda", tool_type: ToolType::Rust });
    registry.insert("gpu", gpu);
    
    // Analysis Tools
    let mut analyze = HashMap::new();
    analyze.insert("harmonics", Tool { bin: "monster_harmonics", tool_type: ToolType::Rust });
    analyze.insert("resonance", Tool { bin: "prime_resonance_hecke", tool_type: ToolType::Rust });
    analyze.insert("semantic", Tool { bin: "analyze_semantic_significance", tool_type: ToolType::Rust });
    registry.insert("analyze", analyze);
    
    // Review Tools
    let mut review = HashMap::new();
    review.insert("multi-level", Tool { bin: "multi_level_review.py", tool_type: ToolType::Python });
    review.insert("paper", Tool { bin: "review_paper", tool_type: ToolType::Rust });
    review.insert("pre-commit", Tool { bin: "pre_commit_review", tool_type: ToolType::Rust });
    registry.insert("review", review);
    
    // Deployment Tools
    let mut deploy = HashMap::new();
    deploy.insert("all", Tool { bin: "deploy_all", tool_type: ToolType::Rust });
    deploy.insert("self", Tool { bin: "self_deploy", tool_type: ToolType::Rust });
    deploy.insert("archive", Tool { bin: "archive_deploy", tool_type: ToolType::Rust });
    registry.insert("deploy", deploy);
    
    registry
}

fn list_tools(registry: &HashMap<&str, HashMap<&str, Tool>>) {
    println!("🔧 Monster Project - Kiro Tool Registry");
    println!("{}", "=".repeat(60));
    
    for (category, tools) in registry.iter() {
        println!("\n{}:", category.to_uppercase());
        for (name, tool) in tools.iter() {
            let type_str = match tool.tool_type {
                ToolType::Rust => "rust",
                ToolType::Python => "python",
            };
            println!("  {:20} ({:6}) -> {}", name, type_str, tool.bin);
        }
    }
    
    println!("\n{}", "=".repeat(60));
    println!("Usage: kiro <category> <tool> [args...]");
    println!("Example: kiro verify monster-walk");
}

fn run_rust_tool(bin_name: &str, args: &[String]) -> i32 {
    let status = Command::new("cargo")
        .args(&["run", "--release", "--bin", bin_name, "--"])
        .args(args)
        .status();
    
    match status {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => {
            eprintln!("❌ Failed to run Rust tool: {}", e);
            1
        }
    }
}

fn run_python_tool(script_name: &str, args: &[String]) -> i32 {
    let status = Command::new("python3")
        .arg(script_name)
        .args(args)
        .status();
    
    match status {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => {
            eprintln!("❌ Failed to run Python tool: {}", e);
            1
        }
    }
}

fn main() {
    let registry = build_registry();
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        list_tools(&registry);
        exit(0);
    }
    
    let cmd = &args[1];
    
    if cmd == "--help" || cmd == "-h" || cmd == "help" || cmd == "list" {
        list_tools(&registry);
        exit(0);
    }
    
    if args.len() < 3 {
        eprintln!("❌ Error: Missing tool name");
        eprintln!("Usage: kiro <category> <tool> [args...]");
        exit(1);
    }
    
    let category = &args[1];
    let tool_name = &args[2];
    let tool_args = &args[3..];
    
    let tools = match registry.get(category.as_str()) {
        Some(t) => t,
        None => {
            eprintln!("❌ Error: Unknown category '{}'", category);
            eprintln!("Available: {}", registry.keys().map(|k| *k).collect::<Vec<_>>().join(", "));
            exit(1);
        }
    };
    
    let tool = match tools.get(tool_name.as_str()) {
        Some(t) => t,
        None => {
            eprintln!("❌ Error: Unknown tool '{}' in category '{}'", tool_name, category);
            eprintln!("Available: {}", tools.keys().map(|k| *k).collect::<Vec<_>>().join(", "));
            exit(1);
        }
    };
    
    let type_str = match tool.tool_type {
        ToolType::Rust => "rust",
        ToolType::Python => "python",
    };
    
    println!("🚀 Running {}/{} ({})...", category, tool_name, type_str);
    
    let code = match tool.tool_type {
        ToolType::Rust => run_rust_tool(tool.bin, tool_args),
        ToolType::Python => run_python_tool(tool.bin, tool_args),
    };
    
    exit(code);
}
