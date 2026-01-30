#!/usr/bin/env python3
# Pipelite: Local GitHub Actions runner with Nix + Archive.org plugin

import subprocess
import sys
import os

def run_cmd(cmd, cwd=None):
    """Run command and stream output"""
    print(f"🔧 {cmd}")
    result = subprocess.run(
        cmd, 
        shell=True, 
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print(result.stdout)
    return result.returncode == 0

def main():
    print("🚀 Pipelite: Local GitHub Actions Runner + Archive.org Plugin")
    print("=" * 70)
    
    repo_root = "/home/mdupont/experiments/monster"
    os.chdir(repo_root)
    
    # Step 0: Test Archive.org plugin
    print("\n🔌 Step 0: Test Archive.org Plugin")
    if not run_cmd("nix develop --command cargo build --release --bin archive_plugin_test", cwd=repo_root):
        print("❌ Plugin build failed")
        return 1
    
    # Step 1: Build WASM Reader
    print("\n📦 Step 1: Build WASM Reader")
    if not run_cmd(
        "nix develop --command wasm-pack build --target web --out-dir pkg",
        cwd=f"{repo_root}/archive_org_reader"
    ):
        print("❌ WASM build failed")
        return 1
    
    run_cmd("mkdir -p archive_org_reader/deploy", cwd=repo_root)
    run_cmd("cp archive_org_reader/index.html archive_org_reader/deploy/", cwd=repo_root)
    run_cmd("cp -r archive_org_reader/pkg archive_org_reader/deploy/", cwd=repo_root)
    
    # Step 2: Build Rust Binaries
    print("\n🔨 Step 2: Build Rust Binaries")
    binaries = [
        "extract_constants",
        "apply_value_lattice", 
        "lattice_qwen_witness",
        "zk_lattice_archive",
        "qwen_to_wasm_hecke",
        "universal_shard_reader",
        "archive_plugin_test"
    ]
    
    for binary in binaries:
        if not run_cmd(f"nix develop --command cargo build --release --bin {binary}", cwd=repo_root):
            print(f"❌ Build failed: {binary}")
            return 1
    
    # Step 3: Generate Artifacts
    print("\n⚙️  Step 3: Generate Artifacts")
    for binary in ["extract_constants", "apply_value_lattice", "lattice_qwen_witness", 
                   "zk_lattice_archive", "qwen_to_wasm_hecke"]:
        run_cmd(f"nix develop --command ./target/release/{binary}", cwd=repo_root)
    
    # Step 4: Test Locally
    print("\n🧪 Step 4: Test Locally")
    print("Starting server at http://localhost:8001")
    print("Press Ctrl+C to stop and continue...")
    
    try:
        subprocess.run(
            "cd archive_org_reader/deploy && python3 -m http.server 8001",
            shell=True,
            cwd=repo_root
        )
    except KeyboardInterrupt:
        print("\n✅ Server stopped")
    
    # Step 5: Upload via Archive.org Plugin
    print("\n📤 Step 5: Upload via Archive.org Plugin")
    
    upload = input("Upload to Archive.org using plugin? (y/N): ")
    if upload.lower() == 'y':
        # Use plugin to upload
        if not run_cmd("nix develop --command ./target/release/archive_plugin_test", cwd=repo_root):
            print("❌ Plugin upload failed")
            return 1
        
        print("\n✅ Upload complete!")
        print("   Data: https://archive.org/details/monster-zk-lattice-v1")
        print("   Reader: https://archive.org/details/monster-zk-lattice-reader")
    
    print("\n✅ Pipelite complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
