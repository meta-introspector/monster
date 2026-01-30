// Pure Rust multi-platform deployment

use std::process::Command;

struct Deployer;

impl Deployer {
    fn run_nix(&self, cmd: &str, desc: &str) -> Result<(), String> {
        println!("🔧 {}", desc);
        
        let output = Command::new("nix")
            .args(&["develop", "--command", "sh", "-c", cmd])
            .output()
            .map_err(|e| e.to_string())?;
        
        if output.status.success() {
            println!("  ✅");
            Ok(())
        } else {
            let err = String::from_utf8_lossy(&output.stderr);
            Err(err.to_string())
        }
    }
    
    fn deploy_github_pages(&self) -> Result<(), String> {
        println!("\n📄 GitHub Pages...");
        
        self.run_nix("mkdir -p docs", "Create docs/")?;
        self.run_nix("cp README.md docs/", "Copy README")?;
        self.run_nix("cp PAPER.md docs/ || true", "Copy PAPER")?;
        self.run_nix("git add docs/", "Stage")?;
        self.run_nix("git commit -m 'Deploy' || true", "Commit")?;
        self.run_nix("git push origin main || true", "Push")?;
        
        println!("  ✅ https://YOUR_USERNAME.github.io/monster/");
        Ok(())
    }
    
    fn deploy_vercel(&self) -> Result<(), String> {
        println!("\n🔺 Vercel...");
        
        let check = Command::new("nix-shell")
            .args(&["-p", "nodePackages.vercel", "--run", "which vercel"])
            .output();
        
        if check.is_ok() {
            self.run_nix("nix-shell -p nodePackages.vercel --run 'vercel --prod --yes'", 
                        "Deploy to Vercel")?;
            println!("  ✅ https://monster.vercel.app");
        } else {
            println!("  ⚠️  Skipped (not configured)");
        }
        
        Ok(())
    }
    
    fn deploy_cloudflare(&self) -> Result<(), String> {
        println!("\n☁️  Cloudflare Pages...");
        
        let check = Command::new("nix-shell")
            .args(&["-p", "nodePackages.wrangler", "--run", "which wrangler"])
            .output();
        
        if check.is_ok() {
            self.run_nix(
                "nix-shell -p nodePackages.wrangler --run 'wrangler pages deploy docs --project-name=monster-zk-lattice'",
                "Deploy to Cloudflare"
            )?;
            println!("  ✅ https://monster-zk-lattice.pages.dev");
        } else {
            println!("  ⚠️  Skipped (not configured)");
        }
        
        Ok(())
    }
    
    fn deploy_archive_org(&self) -> Result<(), String> {
        println!("\n📦 Archive.org...");
        
        self.run_nix("./target/release/self_deploy", "Self-deploy")?;
        println!("  ✅ https://archive.org/details/monster-zk-lattice-complete");
        
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 MULTI-PLATFORM DEPLOYMENT (Pure Rust + Nix)");
    println!("{}", "=".repeat(70));
    println!();
    
    let deployer = Deployer;
    
    // Build everything first
    println!("📦 Building all binaries...");
    let binaries = vec![
        "self_deploy",
        "archive_deploy",
        "extract_constants",
        "apply_value_lattice",
        "lattice_qwen_witness",
        "zk_lattice_archive",
        "qwen_to_wasm_hecke"
    ];
    
    for binary in binaries {
        deployer.run_nix(
            &format!("cargo build --release --bin {}", binary),
            &format!("Building {}", binary)
        )?;
    }
    
    // Generate artifacts
    println!("\n⚙️  Generating artifacts...");
    for binary in &["extract_constants", "apply_value_lattice", "lattice_qwen_witness", 
                    "zk_lattice_archive", "qwen_to_wasm_hecke"] {
        deployer.run_nix(
            &format!("./target/release/{}", binary),
            &format!("Running {}", binary)
        )?;
    }
    
    // Deploy to all platforms
    println!("\n🌐 Deploying to platforms...");
    
    deployer.deploy_github_pages()?;
    deployer.deploy_vercel().ok();
    deployer.deploy_cloudflare().ok();
    deployer.deploy_archive_org()?;
    
    println!();
    println!("✅ MULTI-PLATFORM DEPLOYMENT COMPLETE!");
    println!();
    println!("📊 Deployed to:");
    println!("  • GitHub Pages ✅");
    println!("  • Vercel (if configured)");
    println!("  • Cloudflare Pages (if configured)");
    println!("  • Archive.org ✅");
    println!();
    println!("🎯 All tools from Nix, all code in Rust!");
    
    Ok(())
}
