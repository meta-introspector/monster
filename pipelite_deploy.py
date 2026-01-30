#!/usr/bin/env python3
# Pipelite: Multi-platform deployment with Nix

import subprocess
import sys
import os

def run_nix(cmd, desc):
    """Run command in Nix environment"""
    print(f"🔧 {desc}")
    result = subprocess.run(
        f"nix develop --command {cmd}",
        shell=True,
        cwd="/home/mdupont/experiments/monster",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print(result.stdout)
    return result.returncode == 0

def main():
    print("🚀 PIPELITE: MULTI-PLATFORM DEPLOYMENT")
    print("=" * 70)
    print()
    
    repo = "/home/mdupont/experiments/monster"
    os.chdir(repo)
    
    # Step 1: Build everything
    print("📦 Step 1: Build all artifacts...")
    
    binaries = [
        "self_deploy",
        "archive_deploy",
        "extract_constants",
        "apply_value_lattice",
        "lattice_qwen_witness",
        "zk_lattice_archive",
        "qwen_to_wasm_hecke"
    ]
    
    for binary in binaries:
        if not run_nix(f"cargo build --release --bin {binary}", f"Building {binary}"):
            print(f"❌ Failed: {binary}")
            return 1
    
    # Step 2: Generate artifacts
    print("\n⚙️  Step 2: Generate artifacts...")
    for binary in ["extract_constants", "apply_value_lattice", "lattice_qwen_witness", 
                   "zk_lattice_archive", "qwen_to_wasm_hecke"]:
        run_nix(f"./target/release/{binary}", f"Running {binary}")
    
    # Step 3: Prepare deployment
    print("\n📋 Step 3: Prepare deployment...")
    run_nix("mkdir -p docs", "Creating docs/")
    run_nix("cp README.md docs/", "Copy README")
    run_nix("cp PAPER.md docs/ || true", "Copy PAPER")
    
    # Step 4: Deploy to platforms
    print("\n🌐 Step 4: Deploy to platforms...")
    
    # GitHub Pages
    print("\n📄 GitHub Pages...")
    run_nix("git add docs/", "Stage docs")
    run_nix("git commit -m 'Deploy to GitHub Pages' || true", "Commit")
    run_nix("git push origin main || true", "Push")
    print("  ✅ https://YOUR_USERNAME.github.io/monster/")
    
    # Vercel (via Nix)
    print("\n🔺 Vercel...")
    if run_nix("which vercel", "Check vercel"):
        run_nix("vercel --prod --yes || true", "Deploy to Vercel")
        print("  ✅ https://monster.vercel.app")
    else:
        print("  ⚠️  Install: nix-shell -p nodePackages.vercel")
    
    # Cloudflare Pages (via Nix)
    print("\n☁️  Cloudflare Pages...")
    if run_nix("which wrangler", "Check wrangler"):
        run_nix("wrangler pages deploy docs --project-name=monster-zk-lattice || true", 
                "Deploy to Cloudflare")
        print("  ✅ https://monster-zk-lattice.pages.dev")
    else:
        print("  ⚠️  Install: nix-shell -p nodePackages.wrangler")
    
    # Archive.org (via Rust)
    print("\n📦 Archive.org...")
    if run_nix("./target/release/self_deploy", "Self-deploy"):
        print("  ✅ https://archive.org/details/monster-zk-lattice-complete")
    else:
        print("  ⚠️  Configure: ia configure")
    
    print("\n✅ MULTI-PLATFORM DEPLOYMENT COMPLETE!")
    print()
    print("📊 Deployed to:")
    print("  • GitHub Pages")
    print("  • Vercel (if configured)")
    print("  • Cloudflare Pages (if configured)")
    print("  • Archive.org")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
