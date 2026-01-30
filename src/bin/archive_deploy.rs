// Reusable Archive.org deployment plugin

use std::path::Path;

pub struct ArchivePlugin {
    initialized: bool,
}

impl ArchivePlugin {
    pub fn new() -> Result<Self, String> {
        Ok(ArchivePlugin { initialized: true })
    }

    pub fn upload_item(&self, identifier: &str, file_path: &str, metadata: &str) -> Result<(), String> {
        let mut args = vec!["upload", identifier, file_path];
        
        if !metadata.is_empty() {
            args.push("--metadata");
            args.push(metadata);
        }
        
        let output = std::process::Command::new("ia")
            .args(&args)
            .output()
            .map_err(|e| e.to_string())?;
        
        if output.status.success() {
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        }
    }

    pub fn upload_directory(&self, identifier: &str, dir_path: &str, extension: &str) -> Result<usize, String> {
        let mut count = 0;
        let entries = std::fs::read_dir(dir_path)
            .map_err(|e| e.to_string())?;
        
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            
            if let Some(ext) = path.extension() {
                if extension == "*" || ext.to_str() == Some(extension) {
                    let path_str = path.display().to_string();
                    println!("  Uploading {}", path_str);
                    self.upload_item(identifier, &path_str, "")?;
                    count += 1;
                }
            }
        }
        
        Ok(count)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 3 {
        println!("Usage: {} <identifier> <path1> [path2] [...]", args[0]);
        println!();
        println!("Examples:");
        println!("  {} monster-zk-lattice-v1 archive_org_shards/", args[0]);
        println!("  {} monster-zk-lattice-v1 analysis/value_lattice_witnessed.json", args[0]);
        println!("  {} monster-zk-lattice-v1 wasm_hecke_operators/", args[0]);
        return Ok(());
    }
    
    println!("🔌 ARCHIVE.ORG PLUGIN - DEPLOY");
    println!("{}", "=".repeat(70));
    println!();
    
    let plugin = ArchivePlugin::new()?;
    let identifier = &args[1];
    
    for path in &args[2..] {
        let path_obj = Path::new(path);
        
        if path_obj.is_dir() {
            println!("📂 Uploading directory: {}", path);
            let count = plugin.upload_directory(identifier, path, "*")?;
            println!("  ✅ Uploaded {} files", count);
        } else if path_obj.is_file() {
            println!("📄 Uploading file: {}", path);
            plugin.upload_item(identifier, path, "")?;
            println!("  ✅ Uploaded");
        } else {
            println!("  ⚠️  Skipped {} (not found)", path);
        }
    }
    
    println!();
    println!("✅ Deployment complete!");
    println!("   https://archive.org/details/{}", identifier);
    
    Ok(())
}
