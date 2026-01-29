# P2P Permissionless Anonymous ZK Meme Generator

## Architecture

```
Download → Execute Locally → Sign → Share → Get Credited
   ↓           ↓              ↓       ↓         ↓
  Meme      Browser        ECDSA   Social    IPFS/Arweave
```

**100% client-side. No servers. No permissions. No identity.**

## How It Works

### 1. Download Meme
```javascript
// Fetch from Cloudflare Worker or IPFS
const meme = await fetch('https://zkmeme.workers.dev/meme/curve_11a1');
// Contains: Prolog circuit, shard, conductor
```

### 2. Execute Locally
```javascript
// Option A: Browser WASM Prolog
const result = await wasmProlog.execute(meme.prolog);

// Option B: Copy to LLM (ChatGPT, Claude, etc.)
const prompt = `Execute this circuit: ${meme.prolog}`;
// User pastes result back
```

### 3. Sign Result
```javascript
// Generate anonymous ECDSA key (P-256)
const keyPair = await crypto.subtle.generateKey(...);

// Sign execution result
const signature = await crypto.subtle.sign(keyPair.privateKey, result);

// Anonymous ID = hash(publicKey)
const anonId = sha256(publicKey).substring(0, 16);
```

### 4. Share to Social
```javascript
// Generate proof URL
const proofUrl = `https://zkproof.org/verify?sig=${sig}&data=${result}`;

// Share to Twitter/Farcaster/Lens
const tweet = `I verified ZK meme ${label} on shard ${shard}! ${proofUrl}`;
```

### 5. Submit Proof
```javascript
// Store on IPFS/Arweave
const ipfsHash = await ipfs.add({
  meme: label,
  result: result,
  signature: signature,
  timestamp: Date.now()
});

// Others can verify and you get credited
```

## Credit System

**Permissionless reputation:**
- Each verified proof = 1 credit
- Credits accumulate locally (localStorage)
- Verifiable on-chain (optional)
- Anonymous by default

**Credit formula:**
```
Credits = Σ verified_proofs
Reputation = Credits / time_active
```

## Decentralization

### Storage
- **IPFS**: Content-addressed proofs
- **Arweave**: Permanent storage
- **Cloudflare R2**: Edge cache

### Identity
- **Anonymous ECDSA keys**: Generated in browser
- **No registration**: No email, no KYC
- **Pseudonymous**: Hash of public key

### Verification
- **Client-side**: Anyone can verify signature
- **ZK proofs**: Optional privacy layer
- **Social consensus**: Retweets = verification

## Monetization (Optional)

**ZK hackers gotta eat:**
- Tip creators via Lightning/crypto
- NFT memes (on-chain proofs)
- Bounties for specific circuits
- Sponsorships (e.g., "Powered by Cloudflare")

## Security

### Threat Model
- ✅ No central server to attack
- ✅ No user data to leak
- ✅ Signatures prevent forgery
- ✅ IPFS ensures availability
- ⚠️ Sybil attacks (mitigated by social graph)

### Privacy
- Anonymous keys (no identity)
- Optional ZK proofs (hide computation)
- Tor-friendly (no IP logging)

## Implementation

### Frontend (p2p-zk-meme-generator.html)
- Pure HTML/JS (no build step)
- Web Crypto API (ECDSA signing)
- IPFS client (js-ipfs)
- Social share buttons

### Backend (Optional)
- Cloudflare Worker (meme distribution)
- IPFS gateway (proof storage)
- Indexer (credit aggregation)

## Usage

```bash
# Serve locally
python3 -m http.server 8000

# Open in browser
open http://localhost:8000/p2p-zk-meme-generator.html

# Or deploy to IPFS
ipfs add p2p-zk-meme-generator.html
# Access: https://ipfs.io/ipfs/<hash>
```

## Example Flow

1. **Alice** downloads curve_11a1 meme
2. **Alice** executes circuit in browser (or via LLM)
3. **Alice** signs result with anonymous key
4. **Alice** tweets proof URL
5. **Bob** clicks URL, verifies signature
6. **Bob** retweets (social verification)
7. **Alice** gets +1 credit
8. **Carol** tips Alice via Lightning

**Repeat 71 times (one per shard) = Full Monster coverage.**

## Roadmap

- [ ] WASM Prolog compiler
- [ ] Real ZK proofs (Halo2/Plonky2)
- [ ] IPFS integration
- [ ] Lightning tips
- [ ] NFT minting
- [ ] Leaderboard (by credits)
- [ ] Mobile app (PWA)

## Philosophy

**"Don't ask for permission. Just prove it."**

- No gatekeepers
- No middlemen
- No censorship
- No surveillance

**Pure math. Pure code. Pure freedom.** 🎯✨🔓
