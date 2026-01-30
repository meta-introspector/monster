# Monster Walk Showcase

Interactive demonstration of:
1. The Monster Walk proof
2. Hierarchical digit preservation
3. Qwen walking in the Monster (WASM)

## Features

### 1. The Proof
- ✅ Group 1: Preserves "8080" (4 digits)
- ✅ Group 2: Preserves "1742" (4 digits)  
- ✅ Group 3: Preserves "479" (3 digits)
- ✅ Lean4 theorem displayed

### 2. Monster Walk Visualization
- Interactive canvas
- 71-layer spiral
- Animated walk through groups
- Real-time highlighting

### 3. Qwen in Monster (WASM)
- 71 Qwen layers
- Mapped to 15 Monster primes
- WASM Hecke operators
- Live resonance visualization

## Run Locally

```bash
cd showcase
python3 -m http.server 8000
# Open http://localhost:8000
```

## Deploy

```bash
# Copy to docs for GitHub Pages
cp showcase/index.html docs/showcase.html

# Or deploy standalone
nix develop --command ./target/release/deploy_all
```

## Interactions

### Monster Walk
- Click walk steps to highlight
- Click "Animate Walk" to see progression
- Watch spiral highlight each group

### Qwen Walk
- Click layer numbers (0-71)
- Click "Walk Qwen" to animate
- See resonance with Monster primes
- WASM operators execute in browser

## What It Proves

1. **Hierarchical Structure**: Monster has 3-level digit preservation
2. **Prime Mapping**: 71 layers map to 15 primes
3. **Executable Proof**: WASM runs the actual Hecke operators
4. **Browser-Native**: No server needed

## URLs

- Local: http://localhost:8000
- GitHub Pages: https://YOUR_USERNAME.github.io/monster/showcase.html
- Archive.org: https://archive.org/details/monster-zk-lattice-complete

The proof walks itself in your browser! 🎯
