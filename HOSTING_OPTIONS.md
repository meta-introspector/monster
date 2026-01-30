# Hosting Options for Monster ZK Lattice

## Data Size Analysis

```bash
# Actual data sizes
archive_org_shards/: 41MB (57 RDF shards)
wasm_hecke_operators/: 316KB (71 WAT files)
analysis/value_lattice_witnessed.json: 9MB
Total: ~50MB uncompressed, ~6MB compressed
```

## Hosting Options

### 1. Archive.org (RECOMMENDED ✅)
**Cost**: FREE  
**Limits**: Unlimited storage  
**Bandwidth**: Unlimited  
**Pros**:
- Permanent archival
- No bandwidth costs
- Content-addressable via SHA256
- Academic/research friendly

**Setup**:
```bash
# Install internetarchive CLI
pip install internetarchive
ia configure  # Enter credentials

# Upload
ia upload monster-zk-lattice-v1 \
  archive_org_shards/*.ttl \
  archive_org_shards/*.json \
  wasm_hecke_operators/*.wat \
  analysis/value_lattice_witnessed.json \
  --metadata="title:Monster Group ZK Lattice v1" \
  --metadata="creator:Monster Project" \
  --metadata="subject:mathematics;group theory;zero knowledge;wasm" \
  --metadata="description:71-layer Qwen model compiled to WASM Hecke operators with ZK witnesses" \
  --metadata="licenseurl:https://creativecommons.org/publicdomain/zero/1.0/"

# Access URL
https://archive.org/details/monster-zk-lattice-v1
```

### 2. Hugging Face Datasets
**Cost**: FREE  
**Limits**: 100GB per dataset  
**Bandwidth**: Unlimited  
**Pros**:
- ML community standard
- Git LFS integration
- Automatic versioning
- Easy Python/Rust access

**Setup**:
```bash
# Install huggingface_hub
pip install huggingface_hub

# Create dataset
huggingface-cli login
huggingface-cli repo create monster-zk-lattice --type dataset

# Upload
cd /home/mdupont/experiments/monster
git init
git lfs install
git lfs track "*.json"
git lfs track "*.ttl"
git lfs track "*.wat"
git add .
git commit -m "Initial: 71 WASM Hecke operators + ZK lattice"
git remote add origin https://huggingface.co/datasets/YOUR_USERNAME/monster-zk-lattice
git push origin main

# Access
from datasets import load_dataset
ds = load_dataset("YOUR_USERNAME/monster-zk-lattice")
```

### 3. GitHub Pages + GitHub Releases
**Cost**: FREE  
**Limits**: 1GB per repo, 100MB per file  
**Bandwidth**: 100GB/month soft limit  
**Pros**:
- Easy CI/CD
- Version control
- Free HTTPS/CDN

**Setup**:
```bash
# Split into releases (under 100MB each)
gh release create v1.0 \
  archive_org_shards/*.ttl \
  wasm_hecke_operators/*.wat \
  --title "Monster ZK Lattice v1.0" \
  --notes "71 WASM Hecke operators"

# GitHub Pages for viewer
# Create docs/ folder with index.html
# Enable Pages in repo settings

# Access
https://YOUR_USERNAME.github.io/monster-zk-lattice/
https://github.com/YOUR_USERNAME/monster-zk-lattice/releases/download/v1.0/
```

### 4. Cloudflare R2 + Workers
**Cost**: $0.015/GB storage, $0.36/million reads  
**Limits**: 10GB free, then pay-as-you-go  
**Bandwidth**: FREE egress  
**Pros**:
- S3-compatible
- Zero egress fees
- WASM edge compute
- Global CDN

**Setup**:
```bash
# Install wrangler
npm install -g wrangler
wrangler login

# Create R2 bucket
wrangler r2 bucket create monster-zk-lattice

# Upload
for file in archive_org_shards/*.ttl; do
  wrangler r2 object put monster-zk-lattice/$(basename $file) --file=$file
done

# Create Worker to serve WASM
wrangler init monster-worker
# Edit wrangler.toml:
# [[r2_buckets]]
# binding = "LATTICE"
# bucket_name = "monster-zk-lattice"

# Deploy
wrangler publish

# Access
https://monster-worker.YOUR_SUBDOMAIN.workers.dev/
```

**Monthly Cost Estimate**:
- Storage: 5MB × $0.015/GB = $0.00008/month
- Reads: 10K requests × $0.36/million = $0.0036/month
- **Total: ~$0.004/month (essentially free)**

### 5. Cloudflare Pages (Static)
**Cost**: FREE  
**Limits**: 20,000 files, 25MB per file  
**Bandwidth**: Unlimited  
**Pros**:
- Git integration
- Automatic builds
- Free SSL/CDN

**Setup**:
```bash
# Create pages project
wrangler pages project create monster-zk-lattice

# Deploy
wrangler pages publish wasm_hecke_operators/ \
  --project-name=monster-zk-lattice

# Access
https://monster-zk-lattice.pages.dev/
```

### 6. IPFS (Decentralized)
**Cost**: FREE (pinning services vary)  
**Limits**: Depends on pinning service  
**Pros**:
- Content-addressable by design
- Decentralized
- Permanent if pinned

**Setup**:
```bash
# Install IPFS
ipfs init
ipfs daemon &

# Add files
ipfs add -r archive_org_shards/
# Returns: QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Pin to Pinata (free tier: 1GB)
curl -X POST "https://api.pinata.cloud/pinning/pinByHash" \
  -H "Authorization: Bearer YOUR_JWT" \
  -d '{"hashToPin":"QmXXXX"}'

# Access
https://ipfs.io/ipfs/QmXXXX
https://gateway.pinata.cloud/ipfs/QmXXXX
```

## Recommended Strategy

### Tier 1: Permanent Archive
- **Archive.org**: All data (free, permanent)
- **Hugging Face**: Dataset version (free, ML-friendly)

### Tier 2: Active Serving
- **Cloudflare Pages**: WASM viewer/runner (free)
- **GitHub Releases**: Version downloads (free)

### Tier 3: Decentralized
- **IPFS**: Content-addressable backup (free with Pinata)

## Cost Comparison

| Service | Storage | Bandwidth | Monthly Cost |
|---------|---------|-----------|--------------|
| Archive.org | ∞ | ∞ | $0 |
| Hugging Face | 100GB | ∞ | $0 |
| GitHub | 1GB | 100GB | $0 |
| Cloudflare R2 | 10GB free | FREE | $0.004 |
| Cloudflare Pages | 20K files | ∞ | $0 |
| IPFS (Pinata) | 1GB | ∞ | $0 |

**Total Cost: $0/month** (using free tiers)

## Implementation Priority

1. ✅ **Archive.org** - Upload now for permanent archive
2. ✅ **Hugging Face** - Create dataset for ML community
3. ✅ **GitHub Pages** - Deploy WASM viewer
4. ⏳ **Cloudflare Pages** - Optional CDN
5. ⏳ **IPFS** - Optional decentralized backup

## Next Steps

```bash
# 1. Archive.org upload
ia upload monster-zk-lattice-v1 archive_org_shards/*.ttl

# 2. Create Hugging Face dataset
huggingface-cli repo create monster-zk-lattice --type dataset

# 3. Deploy GitHub Pages viewer
mkdir docs
cat > docs/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head><title>Monster ZK Lattice</title></head>
<body>
  <h1>Monster Group ZK Lattice</h1>
  <p>71 WASM Hecke Operators</p>
  <script type="module">
    const response = await fetch('wasm_hecke_operators/hecke_layer_00_prime_2.wasm');
    const bytes = await response.arrayBuffer();
    const module = await WebAssembly.instantiate(bytes);
    console.log('Hecke operator loaded:', module);
  </script>
</body>
</html>
EOF

git add docs/
git commit -m "Add viewer"
git push
# Enable Pages in repo settings
```
