# Testing and Deployment Guide

## Local Testing with Pipelite

### Quick Test
```bash
cd /home/mdupont/experiments/monster

# Run pipelite (interactive)
python3 pipelite.py
```

### Manual Steps
```bash
# 1. Build WASM
cd archive_org_reader
nix develop --command wasm-pack build --target web --out-dir pkg

# 2. Test locally
cd deploy
python3 -m http.server 8001
# Open http://localhost:8001

# 3. Build all binaries
cd /home/mdupont/experiments/monster
nix develop --command cargo build --release --bin extract_constants
nix develop --command cargo build --release --bin apply_value_lattice
nix develop --command cargo build --release --bin lattice_qwen_witness
nix develop --command cargo build --release --bin zk_lattice_archive
nix develop --command cargo build --release --bin qwen_to_wasm_hecke

# 4. Generate artifacts
nix develop --command ./target/release/extract_constants
nix develop --command ./target/release/apply_value_lattice
nix develop --command ./target/release/lattice_qwen_witness
nix develop --command ./target/release/zk_lattice_archive
nix develop --command ./target/release/qwen_to_wasm_hecke
```

## Upload to Archive.org

### Setup
```bash
# Install internetarchive
nix develop --command pip install internetarchive

# Configure (one-time)
nix develop --command ia configure
# Enter email and password
```

### Upload
```bash
# Upload data
nix develop --command ia upload monster-zk-lattice-v1 \
  archive_org_shards/*.ttl \
  archive_org_shards/*.json \
  analysis/value_lattice_witnessed.json \
  wasm_hecke_operators/*.wat \
  --metadata="title:Monster ZK Lattice Data v1" \
  --metadata="creator:Monster Project" \
  --metadata="subject:mathematics;group theory;zero knowledge"

# Upload reader
nix develop --command ia upload monster-zk-lattice-reader \
  archive_org_reader/deploy/index.html \
  archive_org_reader/deploy/pkg/*.js \
  archive_org_reader/deploy/pkg/*.wasm \
  --metadata="title:Monster ZK Lattice WASM Reader" \
  --metadata="creator:Monster Project"
```

## GitHub Actions (Self-Hosted Runner)

### Setup Runner
```bash
# Install GitHub Actions runner
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# Configure
./config.sh --url https://github.com/YOUR_USERNAME/monster --token YOUR_TOKEN

# Run
./run.sh
```

### Add Secrets
Go to GitHub repo → Settings → Secrets → Actions:
- `IA_ACCESS_KEY`: Archive.org email
- `IA_SECRET_KEY`: Archive.org password

### Trigger Workflow
```bash
# Push to main
git add .
git commit -m "Trigger build"
git push origin main

# Or manual trigger
gh workflow run archive_upload.yml
```

## Pipelite Features

✅ **Local GitHub Actions**: Run workflows locally  
✅ **Nix Integration**: All tools from Nix  
✅ **Interactive**: Prompts for upload  
✅ **Streaming Output**: See progress in real-time  

## File Structure

```
.github/workflows/archive_upload.yml  - GitHub Actions workflow
pipelite.py                           - Local runner
test_and_upload.sh                    - Bash script
archive_org_reader/
  build.sh                            - WASM build
  deploy/                             - Deployment artifacts
```

## URLs After Upload

**Archive.org**:
- Data: https://archive.org/details/monster-zk-lattice-v1
- Reader: https://archive.org/details/monster-zk-lattice-reader

**GitHub Pages** (if enabled):
- https://YOUR_USERNAME.github.io/monster/reader/

## Testing Checklist

- [ ] Build WASM locally
- [ ] Test reader at http://localhost:8001
- [ ] Build all Rust binaries
- [ ] Generate all artifacts
- [ ] Upload to Archive.org
- [ ] Verify data accessible
- [ ] Verify reader works with remote data
- [ ] Test GitHub Actions workflow
- [ ] Deploy to GitHub Pages

## Troubleshooting

**WASM build fails**:
```bash
nix develop --command cargo install wasm-pack
```

**ia command not found**:
```bash
nix develop --command pip install internetarchive
```

**GitHub Actions runner offline**:
```bash
cd ~/actions-runner
./run.sh
```

## Complete Pipeline

```bash
# 1. Local test
python3 pipelite.py

# 2. Commit and push
git add .
git commit -m "Update artifacts"
git push

# 3. GitHub Actions runs automatically
# 4. Artifacts uploaded to Archive.org
# 5. Reader deployed to GitHub Pages
```

All tools from Nix, all data on Archive.org! 🎯
