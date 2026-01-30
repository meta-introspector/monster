#!/usr/bin/env bash
# Deploy Monster Project to Archive.org - Complete Package

set -e

echo "📦 MONSTER PROJECT - COMPLETE DEPLOYMENT"
echo "========================================================================"
echo ""

cd /home/mdupont/experiments/monster

# Step 1: Build everything
echo "🔨 Step 1: Building all artifacts..."
nix develop --command cargo build --release --bin archive_plugin_test
nix develop --command cargo build --release --bin extract_constants
nix develop --command cargo build --release --bin apply_value_lattice
nix develop --command cargo build --release --bin lattice_qwen_witness
nix develop --command cargo build --release --bin zk_lattice_archive
nix develop --command cargo build --release --bin qwen_to_wasm_hecke

# Generate artifacts
echo "⚙️  Generating artifacts..."
nix develop --command ./target/release/extract_constants
nix develop --command ./target/release/apply_value_lattice
nix develop --command ./target/release/lattice_qwen_witness
nix develop --command ./target/release/zk_lattice_archive
nix develop --command ./target/release/qwen_to_wasm_hecke

# Build WASM
echo "🌐 Building WASM reader..."
cd archive_org_reader
nix develop --command wasm-pack build --target web --out-dir pkg
mkdir -p deploy
cp index.html deploy/
cp -r pkg deploy/
cd ..

# Step 2: Collect all files
echo ""
echo "📋 Step 2: Collecting files for upload..."

mkdir -p deploy_package

# Documentation
echo "  📄 Documentation..."
cp README.md deploy_package/
cp PAPER.md deploy_package/ 2>/dev/null || echo "No PAPER.md"
cp *.md deploy_package/ 2>/dev/null || true

# Data
echo "  💾 Data..."
cp -r archive_org_shards deploy_package/
cp -r analysis deploy_package/
cp -r wasm_hecke_operators deploy_package/

# WASM Reader
echo "  🌐 WASM Reader..."
cp -r archive_org_reader/deploy deploy_package/wasm_reader

# Source code
echo "  💻 Source..."
mkdir -p deploy_package/src
cp -r src/bin deploy_package/src/
cp Cargo.toml deploy_package/

# Lean4 proofs
echo "  📐 Lean4 Proofs..."
cp -r MonsterLean deploy_package/ 2>/dev/null || true

# MiniZinc models
echo "  🔢 MiniZinc..."
cp -r minizinc deploy_package/ 2>/dev/null || true

# Images (if any)
echo "  🖼️  Images..."
find . -name "*.png" -o -name "*.jpg" -o -name "*.svg" | head -20 | while read img; do
    cp "$img" deploy_package/ 2>/dev/null || true
done

# Audio (if any)
echo "  🎵 Audio..."
find . -name "*.wav" -o -name "*.mp3" | head -10 | while read audio; do
    cp "$audio" deploy_package/ 2>/dev/null || true
done

# Step 3: Create manifest
echo ""
echo "📝 Step 3: Creating manifest..."

cat > deploy_package/MANIFEST.md << 'EOF'
# Monster Group ZK Lattice - Complete Package

## Contents

### Documentation
- README.md - Project overview
- PAPER.md - Complete paper
- *.md - All documentation files

### Data
- archive_org_shards/ - 57 RDF shards (41MB)
- analysis/ - Value lattice (9MB)
- wasm_hecke_operators/ - 71 WASM operators (316KB)

### Code
- src/bin/ - Rust implementations
- MonsterLean/ - Lean4 proofs
- minizinc/ - Constraint models
- wasm_reader/ - Interactive WASM reader

### Artifacts
- Value lattice with 71,000 ZK witnesses
- 71 Hecke operators compiled to WASM
- RDF semantic shards
- Content-addressable hashes

## Usage

1. Extract package
2. Read README.md
3. Open wasm_reader/index.html in browser
4. Explore data in archive_org_shards/

## Links

- GitHub: https://github.com/meta-introspector/monster-lean
- Archive.org: https://archive.org/details/monster-zk-lattice-complete
- Paper: PAPER.md

## License

CC0 1.0 Universal (Public Domain)
EOF

# Step 4: Upload via plugin
echo ""
echo "📤 Step 4: Uploading to Archive.org..."

# Create metadata
METADATA="title:Monster Group ZK Lattice - Complete Package"
METADATA="$METADATA;creator:Monster Project"
METADATA="$METADATA;subject:mathematics;group theory;zero knowledge;wasm;lean4"
METADATA="$METADATA;description:Complete Monster Group ZK Lattice project with data, code, proofs, and WASM reader"
METADATA="$METADATA;date:$(date -I)"
METADATA="$METADATA;language:eng"
METADATA="$METADATA;licenseurl:https://creativecommons.org/publicdomain/zero/1.0/"

# Upload complete package
echo "  Uploading complete package..."
nix develop --command ia upload monster-zk-lattice-complete \
    deploy_package/* \
    --metadata="$METADATA" \
    --no-derive

# Upload just the reader (for easy access)
echo "  Uploading WASM reader..."
nix develop --command ia upload monster-zk-lattice-reader \
    archive_org_reader/deploy/index.html \
    archive_org_reader/deploy/pkg/*.js \
    archive_org_reader/deploy/pkg/*.wasm \
    --metadata="title:Monster ZK Lattice WASM Reader" \
    --metadata="creator:Monster Project" \
    --metadata="subject:wasm;interactive" \
    --no-derive

echo ""
echo "✅ DEPLOYMENT COMPLETE!"
echo ""
echo "📦 Archive.org URLs:"
echo "   Complete Package: https://archive.org/details/monster-zk-lattice-complete"
echo "   WASM Reader: https://archive.org/details/monster-zk-lattice-reader"
echo ""
echo "📊 Package Contents:"
du -sh deploy_package
echo ""
echo "🎯 Self-Deployed: The project deployed itself to Archive.org!"
