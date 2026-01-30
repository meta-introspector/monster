#!/usr/bin/env bash
# Test locally, build with Nix, upload to Archive.org

set -e

echo "🔨 Building WASM with Nix..."
cd /home/mdupont/experiments/monster/archive_org_reader

nix develop --command wasm-pack build --target web --out-dir pkg

echo "📦 Creating deployment..."
mkdir -p deploy
cp index.html deploy/
cp -r pkg deploy/

echo "🧪 Testing locally..."
cd deploy
python3 -m http.server 8001 &
SERVER_PID=$!
sleep 2

echo "✅ Server running at http://localhost:8001"
echo "   Press Ctrl+C to stop and continue to upload"

wait $SERVER_PID || true

echo ""
echo "📤 Uploading to Archive.org..."
cd /home/mdupont/experiments/monster

# Upload data shards
nix develop --command ia upload monster-zk-lattice-v1 \
  archive_org_shards/*.ttl \
  archive_org_shards/*.json \
  analysis/value_lattice_witnessed.json \
  --metadata="title:Monster ZK Lattice Data v1" \
  --metadata="creator:Monster Project" \
  --metadata="subject:mathematics;group theory;zero knowledge" \
  --metadata="licenseurl:https://creativecommons.org/publicdomain/zero/1.0/"

# Upload WASM reader
nix develop --command ia upload monster-zk-lattice-reader \
  archive_org_reader/deploy/index.html \
  archive_org_reader/deploy/pkg/*.js \
  archive_org_reader/deploy/pkg/*.wasm \
  --metadata="title:Monster ZK Lattice WASM Reader" \
  --metadata="creator:Monster Project" \
  --metadata="subject:wasm;mathematics;interactive"

echo ""
echo "✅ Complete!"
echo "   Data: https://archive.org/details/monster-zk-lattice-v1"
echo "   Reader: https://archive.org/details/monster-zk-lattice-reader"
