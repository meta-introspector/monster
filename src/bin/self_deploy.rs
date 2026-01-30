// Complete Monster Project Uploader - Self-Deploy Everything

use std::path::Path;
use std::fs;

pub struct ArchivePlugin {
    initialized: bool,
}

impl ArchivePlugin {
    pub fn new() -> Result<Self, String> {
        Ok(ArchivePlugin { initialized: true })
    }

    pub fn upload_item(&self, identifier: &str, file_path: &str) -> Result<(), String> {
        let output = std::process::Command::new("ia")
            .args(&["upload", identifier, file_path])
            .output()
            .map_err(|e| e.to_string())?;
        
        if output.status.success() {
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        }
    }

    pub fn upload_directory_recursive(&self, identifier: &str, dir_path: &str) -> Result<usize, String> {
        let mut count = 0;
        
        for entry in fs::read_dir(dir_path).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            
            if path.is_file() {
                let path_str = path.display().to_string();
                println!("  📄 {}", path_str);
                self.upload_item(identifier, &path_str)?;
                count += 1;
            } else if path.is_dir() {
                count += self.upload_directory_recursive(identifier, &path.display().to_string())?;
            }
        }
        
        Ok(count)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 MONSTER PROJECT - COMPLETE SELF-DEPLOYMENT");
    println!("{}", "=".repeat(70));
    println!();
    
    let plugin = ArchivePlugin::new()?;
    let identifier = "monster-zk-lattice-complete";
    
    let mut total_files = 0;
    
    // 1. Documentation
    println!("📚 Uploading documentation...");
    for entry in fs::read_dir(".")? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("md") {
            let path_str = path.display().to_string();
            println!("  📄 {}", path_str);
            plugin.upload_item(identifier, &path_str)?;
            total_files += 1;
        }
    }
    
    // 2. Data directories
    let data_dirs = vec![
        "archive_org_shards",
        "analysis",
        "wasm_hecke_operators",
    ];
    
    for dir in data_dirs {
        if Path::new(dir).exists() {
            println!();
            println!("📂 Uploading {}...", dir);
            let count = plugin.upload_directory_recursive(identifier, dir)?;
            println!("  ✅ {} files", count);
            total_files += count;
        }
    }
    
    // 3. Source code
    println!();
    println!("💻 Uploading source code...");
    plugin.upload_item(identifier, "Cargo.toml")?;
    total_files += 1;
    
    if Path::new("src/bin").exists() {
        let count = plugin.upload_directory_recursive(identifier, "src/bin")?;
        println!("  ✅ {} source files", count);
        total_files += count;
    }
    
    // 4. Proofs
    if Path::new("MonsterLean").exists() {
        println!();
        println!("📐 Uploading Lean4 proofs...");
        let count = plugin.upload_directory_recursive(identifier, "MonsterLean")?;
        println!("  ✅ {} proof files", count);
        total_files += count;
    }
    
    // 5. Models
    if Path::new("minizinc").exists() {
        println!();
        println!("🔢 Uploading MiniZinc models...");
        let count = plugin.upload_directory_recursive(identifier, "minizinc")?;
        println!("  ✅ {} model files", count);
        total_files += count;
    }
    
    // 6. WASM reader
    if Path::new("archive_org_reader/deploy").exists() {
        println!();
        println!("🌐 Uploading WASM reader...");
        let count = plugin.upload_directory_recursive(identifier, "archive_org_reader/deploy")?;
        println!("  ✅ {} reader files", count);
        total_files += count;
    }
    
    println!();
    println!("✅ COMPLETE SELF-DEPLOYMENT FINISHED!");
    println!("{}", "=".repeat(70));
    println!("📊 Total files uploaded: {}", total_files);
    println!("🌐 URL: https://archive.org/details/{}", identifier);
    println!();
    println!("🎯 The Monster Project has deployed itself!");
    
    Ok(())
}
