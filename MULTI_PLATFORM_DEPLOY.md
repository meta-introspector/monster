# Multi-Platform Deployment Guide

## Platforms

The Monster Project deploys to 5 platforms simultaneously:

### 1. GitHub Pages (Free)
- **URL**: https://YOUR_USERNAME.github.io/monster/
- **Content**: WASM reader + docs
- **Setup**: Enable Pages in repo settings

### 2. Vercel (Free)
- **URL**: https://monster.vercel.app
- **Content**: Static site + WASM
- **Setup**: `npm i -g vercel && vercel login`

### 3. Cloudflare Workers (Free)
- **URL**: https://monster-zk-lattice.workers.dev
- **Content**: WASM + R2 data storage
- **Setup**: `npm i -g wrangler && wrangler login`

### 4. Cloudflare Pages (Free)
- **URL**: https://monster-zk-lattice.pages.dev
- **Content**: Static site
- **Setup**: Same as Workers

### 5. Archive.org (Free)
- **URL**: https://archive.org/details/monster-zk-lattice-complete
- **Content**: Complete project archive
- **Setup**: `ia configure`

## Quick Deploy

```bash
# All platforms at once
./deploy_multiplatform.sh
```

## Individual Deploys

### GitHub Pages
```bash
mkdir -p docs
cp -r archive_org_reader/deploy/* docs/
git add docs/
git commit -m "Deploy"
git push
```

### Vercel
```bash
vercel --prod
```

### Cloudflare Pages
```bash
wrangler pages deploy docs --project-name=monster-zk-lattice
```

### Cloudflare Workers
```bash
cd cloudflare-worker
wrangler deploy
```

### Archive.org
```bash
nix develop --command ./target/release/self_deploy
```

## GitHub Actions

Push to trigger automatic deployment:
```bash
git push origin main
```

## Platform Comparison

| Platform | Cost | Speed | Storage | WASM |
|----------|------|-------|---------|------|
| GitHub Pages | Free | Fast | 1GB | ✅ |
| Vercel | Free | Fastest | 100GB | ✅ |
| CF Workers | Free | Fastest | R2 | ✅ |
| CF Pages | Free | Fastest | 20K files | ✅ |
| Archive.org | Free | Slow | ∞ | ✅ |

## URLs After Deployment

All platforms serve the same content:
- Interactive WASM reader
- Monster ZK Lattice data
- Documentation
- Source code links

## Self-Deployment

The project deploys itself to all platforms using:
1. Rust binary (`self_deploy`)
2. Bash script (`deploy_multiplatform.sh`)
3. GitHub Actions (`.github/workflows/multi_deploy.yml`)

## Verification

After deployment, verify each platform:
```bash
curl https://YOUR_USERNAME.github.io/monster/
curl https://monster.vercel.app
curl https://monster-zk-lattice.pages.dev
curl https://monster-zk-lattice.workers.dev
curl https://archive.org/details/monster-zk-lattice-complete
```

All platforms deployed! 🎯
