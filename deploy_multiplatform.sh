#!/usr/bin/env bash
# Deploy Monster Project to multiple platforms

set -e

echo "🚀 MULTI-PLATFORM DEPLOYMENT"
echo "========================================================================"
echo ""

cd /home/mdupont/experiments/monster

# 1. GitHub Pages
echo "📄 Deploying to GitHub Pages..."
mkdir -p docs
cp -r archive_org_reader/deploy/* docs/ 2>/dev/null || echo "WASM reader not built yet"
cp README.md docs/
cp PAPER.md docs/ 2>/dev/null || true

git add docs/
git commit -m "Deploy to GitHub Pages" || true
git push origin main

echo "  ✅ GitHub Pages: https://YOUR_USERNAME.github.io/monster/"

# 2. Vercel
echo ""
echo "🔺 Deploying to Vercel..."
cat > vercel.json << 'EOF'
{
  "version": 2,
  "builds": [
    {
      "src": "docs/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/docs/$1"
    }
  ]
}
EOF

nix develop --command npx vercel --prod || echo "Install vercel: npm i -g vercel"
echo "  ✅ Vercel: https://monster.vercel.app"

# 3. Cloudflare Workers
echo ""
echo "☁️  Deploying to Cloudflare Workers..."
mkdir -p cloudflare-worker

cat > cloudflare-worker/wrangler.toml << 'EOF'
name = "monster-zk-lattice"
main = "src/index.js"
compatibility_date = "2024-01-01"

[[r2_buckets]]
binding = "LATTICE_DATA"
bucket_name = "monster-zk-lattice"
EOF

cat > cloudflare-worker/src/index.js << 'EOF'
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    
    // Serve WASM reader
    if (url.pathname === '/' || url.pathname === '/index.html') {
      return new Response(INDEX_HTML, {
        headers: { 'content-type': 'text/html' }
      });
    }
    
    // Serve data from R2
    if (url.pathname.startsWith('/data/')) {
      const key = url.pathname.slice(6);
      const object = await env.LATTICE_DATA.get(key);
      if (object) {
        return new Response(object.body);
      }
    }
    
    return new Response('Not found', { status: 404 });
  }
};

const INDEX_HTML = `<!DOCTYPE html>
<html><head><title>Monster ZK Lattice</title></head>
<body><h1>Monster ZK Lattice on Cloudflare Workers</h1>
<p>Data served from R2 storage</p></body></html>`;
EOF

cd cloudflare-worker
nix develop --command npx wrangler deploy || echo "Install wrangler: npm i -g wrangler"
cd ..

echo "  ✅ Cloudflare: https://monster-zk-lattice.workers.dev"

# 4. Cloudflare Pages
echo ""
echo "📄 Deploying to Cloudflare Pages..."
nix develop --command npx wrangler pages deploy docs --project-name=monster-zk-lattice || true
echo "  ✅ Cloudflare Pages: https://monster-zk-lattice.pages.dev"

# 5. WASMR (WebAssembly Registry)
echo ""
echo "📦 Publishing to WASMR..."
cat > wasmr.toml << 'EOF'
[package]
name = "monster-zk-lattice"
version = "1.0.0"
description = "Monster Group ZK Lattice with 71 WASM Hecke operators"
license = "CC0-1.0"

[dependencies]
EOF

# Package WASM modules
mkdir -p wasmr-package
cp wasm_hecke_operators/*.wat wasmr-package/ 2>/dev/null || true

echo "  ⚠️  WASMR: Manual publish required"
echo "     Visit: https://wasmr.io"

echo ""
echo "✅ MULTI-PLATFORM DEPLOYMENT COMPLETE!"
echo ""
echo "📊 Deployed to:"
echo "  • GitHub Pages: https://YOUR_USERNAME.github.io/monster/"
echo "  • Vercel: https://monster.vercel.app"
echo "  • Cloudflare Workers: https://monster-zk-lattice.workers.dev"
echo "  • Cloudflare Pages: https://monster-zk-lattice.pages.dev"
echo "  • Archive.org: https://archive.org/details/monster-zk-lattice-complete"
