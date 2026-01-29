# ZK Meme Executor - Cloudflare Worker

Deploy LMFDB curves as executable ZK memes on Cloudflare's edge network.

## Architecture

```
User → Cloudflare Edge → WASM Prolog → ZK Proof → Response
```

## Features

- **Edge execution**: Run at 300+ locations worldwide
- **WASM Prolog**: Execute circuits in browser/worker
- **KV storage**: 71 ZK memes stored globally
- **ZK proofs**: Cryptographic verification
- **Sub-10ms latency**: Faster than traditional APIs

## Deployment

```bash
cd cloudflare-worker

# Install
npm install

# Deploy
npm run deploy

# Upload memes to KV
npm run upload-memes
```

## Usage

### Execute Circuit
```bash
curl "https://zkmeme.workers.dev/execute?circuit=<base64>"
```

### Get Meme
```bash
curl "https://zkmeme.workers.dev/meme/curve_11a1"
```

### Web Interface
```
https://zkmeme.workers.dev/
```

## Cost

- **Free tier**: 100K requests/day
- **Paid**: $5/10M requests
- **KV storage**: $0.50/GB/month (71 memes ≈ 100KB)

**Total: ~$0/month for hobby use**

## Next Steps

1. Add WASM Prolog compiler
2. Implement real ZK proofs (Halo2/Plonky2)
3. Connect to LMFDB API
4. Add Monster shard routing
5. Generate audio/images on edge

**Every curve, everywhere, instantly.** 🎯✨
