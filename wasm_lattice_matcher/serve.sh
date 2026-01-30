#!/bin/bash
# Build and serve WASM lattice matcher

set -e

cd wasm_lattice_matcher

echo "🔨 Building WASM module..."
wasm-pack build --target web --out-dir pkg

echo "🌐 Starting local server..."
echo "Open: http://localhost:8000"
python3 -m http.server 8000
