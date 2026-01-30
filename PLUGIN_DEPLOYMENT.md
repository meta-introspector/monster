# Complete Testing & Deployment with Archive.org Plugin

## Setup

```bash
cd /home/mdupont/experiments/monster

# Install internetarchive in Nix environment
nix develop --command pip install --user internetarchive

# Configure (one-time)
nix develop --command ia configure
```

## Test Archive.org Plugin

```bash
# Build plugin
nix develop --command cargo build --release --bin archive_plugin_test

# Test (dry run)
nix develop --command ./target/release/archive_plugin_test
```

## Run Pipelite

```bash
# Full pipeline with plugin
python3 pipelite.py
```

## Pipeline Steps

1. **Build Archive.org Plugin** ✅
2. **Build WASM Reader** (wasm-pack)
3. **Build All Rust Binaries** (6 programs)
4. **Generate Artifacts** (lattice, shards, WASM operators)
5. **Test Locally** (http://localhost:8001)
6. **Upload via Plugin** (Archive.org)

## Archive.org Plugin Features

From `zos-server/src/extra_plugins/foundation_plugins.rs`:

```rust
pub struct ArchivePlugin {
    pub fn search_archive(&self, query: &str) -> Result<String, String>
    pub fn download_item(&self, identifier: &str, local_path: &str) -> Result<(), String>
    pub fn upload_item(&self, identifier: &str, file_path: &str, metadata: &str) -> Result<(), String>
}
```

## Usage

### Via Pipelite
```bash
python3 pipelite.py
# Follow prompts
# Upload via plugin: y
```

### Via Plugin Directly
```bash
nix develop --command ./target/release/archive_plugin_test
```

### Via GitHub Actions
```bash
git add .
git commit -m "Deploy with plugin"
git push
# Self-hosted runner executes workflow
```

## GitHub Actions Integration

`.github/workflows/archive_upload.yml` now includes:

```yaml
- name: Build Archive.org Plugin
  run: nix develop --command cargo build --release --bin archive_plugin_test

- name: Upload via Plugin
  run: nix develop --command ./target/release/archive_plugin_test
```

## All Binaries from Nix

✅ `wasm-pack` - WASM build  
✅ `cargo` - Rust build  
✅ `ia` - Archive.org CLI (via pip in nix)  
✅ `python3` - Pipelite runner  
✅ Archive.org plugin - Custom Rust integration  

## URLs After Deploy

- **Data**: https://archive.org/details/monster-zk-lattice-v1
- **Reader**: https://archive.org/details/monster-zk-lattice-reader
- **GitHub Pages**: https://YOUR_USERNAME.github.io/monster/reader/

## Foundation Plugins Architecture

```
Foundation Layer (from zos-server)
  ├── Archive.org Plugin (historical records)
  ├── LMFDB Plugin (mathematical data)
  ├── Wikidata Plugin (semantic meaning)
  ├── OSM Plugin (location data)
  └── SDF.org Plugin (community)

Monster Project Integration
  ├── Archive.org Plugin ✅
  ├── Value Lattice
  ├── ZK Witnesses
  ├── WASM Hecke Operators
  └── RDF Shards
```

## Complete Flow

```
Local Development
  ↓
Archive.org Plugin (Rust)
  ↓
Pipelite (Python + Nix)
  ↓
Build All (WASM + Rust)
  ↓
Test Local (http://localhost:8001)
  ↓
Upload (ia CLI via plugin)
  ↓
Archive.org (permanent storage)
  ↓
GitHub Actions (CI/CD)
  ↓
GitHub Pages (deployment)
```

Everything runs through Nix, everything uses the Archive.org plugin! 🎯
