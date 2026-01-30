#!/bin/bash
# Build and deploy Archive.org reader

set -e

cd archive_org_reader

echo "🔨 Building WASM..."
wasm-pack build --target web --out-dir pkg

echo "📦 Creating deployment package..."
mkdir -p deploy
cp index.html deploy/
cp -r pkg deploy/

echo "📝 Creating upload script..."
cat > deploy/upload_to_archive.sh << 'EOF'
#!/bin/bash
# Upload to Archive.org

ia upload monster-zk-lattice-reader \
  index.html \
  pkg/*.js \
  pkg/*.wasm \
  --metadata="title:Monster ZK Lattice Reader (WASM)" \
  --metadata="creator:Monster Project" \
  --metadata="subject:wasm;mathematics;group theory" \
  --metadata="description:WASM-based reader for Monster ZK Lattice shards hosted on Archive.org"

echo "✅ Uploaded to: https://archive.org/details/monster-zk-lattice-reader"
EOF

chmod +x deploy/upload_to_archive.sh

echo ""
echo "✅ Build complete!"
echo ""
echo "📂 Files in deploy/:"
ls -lh deploy/
echo ""
echo "🚀 Next steps:"
echo "  1. Test locally: cd deploy && python3 -m http.server 8000"
echo "  2. Upload to Archive.org: cd deploy && ./upload_to_archive.sh"
echo "  3. Access at: https://archive.org/details/monster-zk-lattice-reader"
