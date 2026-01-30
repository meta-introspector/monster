# How to Use the Monster Lattice Matcher

## Quick Start (5 minutes)

### 1. Build the WASM Module

```bash
cd /home/mdupont/experiments/monster/wasm_lattice_matcher

# Install wasm-pack (one-time)
cargo install wasm-pack

# Build WASM
wasm-pack build --target web --out-dir pkg
```

### 2. Start Local Server

```bash
# Serve the app
python3 -m http.server 8000

# Open in browser
http://localhost:8000
```

### 3. Use the Interface

**Enter a Value**:
- Type `24` in the input box
- Click "Match"
- See: "Layer 0: Prime 2 (resonance: 1.00)"

**Scan All Layers**:
- Type `71`
- Click "Scan All 71 Layers"
- See all layers where 71 resonates

**Visualize**:
- Type any value
- Click "Visualize"
- See bar chart + 71-layer spiral

## What Each Button Does

### "Match" Button
- Finds layers with resonance > 0.5
- Shows: Layer number, Prime, Resonance score
- Example: Value `24` matches layers 0,1 (primes 2,3)

### "Scan All 71 Layers" Button
- Checks resonance across all 71 layers
- Highlights matches in yellow
- Shows which Monster primes divide your value

### "Visualize" Button
- Draws bar chart (15 primes)
- Draws 71-layer spiral
- Yellow dots = resonant layers

### Layer Slider
- Drag to navigate layers 0-70
- Shows current prime for that layer
- Updates display in real-time

## Understanding the Display

### Bar Chart (Top)
```
Height = Resonance strength (0-1)
Color = Green (high) → Red (low)
Labels = Monster primes (2,3,5,7,11...)
```

### Spiral (Center)
```
71 layers in 5-turn spiral
Green line = Layer connections
Yellow dots = Resonant layers (>0.5)
```

## Example Workflows

### Find Layers for Value 24
1. Enter `24`
2. Click "Scan All 71 Layers"
3. Result: Layers 0,1,15,16,30,31,45,46,60,61 (primes 2,3)

### Explore Prime 71
1. Enter `71`
2. Click "Visualize"
3. See: Only layers 14,29,44,59 light up (prime 71)

### Navigate Layers
1. Enter `23`
2. Drag slider to layer 8
3. See: "Prime: 23" (perfect match!)

### Compare Values
1. Enter `24`, click "Visualize"
2. Enter `71`, click "Visualize"
3. Compare spiral patterns

## Resonance Scores Explained

**1.0** = Perfect resonance (value divisible by prime)
- Example: Value 24, Prime 2 → 24 % 2 = 0 → Score 1.0

**0.5-0.99** = Strong resonance (small remainder)
- Example: Value 25, Prime 2 → 25 % 2 = 1 → Score 0.5

**0.0-0.49** = Weak resonance (large remainder)
- Example: Value 24, Prime 71 → 24 % 71 = 24 → Score 0.04

## Advanced Usage

### Load Custom Value Lattice

Edit `index.html`, line 60:
```javascript
const values = [
    ["24", 155],  // [value, gödel_number]
    ["71", 21],
    // Add your values here
];
```

### Export Resonance Data

Open browser console:
```javascript
const field = matcher.compute_resonance_field("24");
console.log(field);  // [1.0, 1.0, 0.2, ...]
```

### Integrate with Your App

```javascript
import init, { LatticeMatcher } from './pkg/wasm_lattice_matcher.js';

await init();
const matcher = new LatticeMatcher();
matcher.add_value("24", 155);

const matches = matcher.find_matches("24");
console.log(matches);
```

## Deploy to Web

### GitHub Pages
```bash
cd /home/mdupont/experiments/monster

# Copy to docs/
mkdir -p docs/lattice-matcher
cp -r wasm_lattice_matcher/* docs/lattice-matcher/

git add docs/
git commit -m "Add lattice matcher"
git push

# Enable Pages in repo settings
# Access: https://YOUR_USERNAME.github.io/monster/lattice-matcher/
```

### Cloudflare Pages
```bash
cd wasm_lattice_matcher
wrangler pages publish . --project-name=monster-lattice-matcher

# Access: https://monster-lattice-matcher.pages.dev/
```

## Troubleshooting

**WASM not loading?**
- Check browser console for errors
- Ensure `pkg/` directory exists
- Try: `wasm-pack build --target web`

**Canvas not drawing?**
- Enter a value first
- Click "Visualize"
- Check if value is numeric

**No matches found?**
- Try values divisible by small primes: 2,3,5,7,11
- Example: 24, 30, 42, 60, 71

## What's Happening Under the Hood

1. **Input**: You enter "24"
2. **WASM**: Rust code computes `24 % prime` for each prime
3. **Resonance**: Converts remainder to score (0-1)
4. **Display**: JavaScript draws bars + spiral
5. **Self-Modify**: Canvas redraws on every change

## Next Steps

- Load actual Qwen weights from `analysis/value_lattice_witnessed.json`
- Add RDF shard integration from `archive_org_shards/`
- Connect to live WASM Hecke operators from `wasm_hecke_operators/`
- Deploy to archive.org + Hugging Face

## Summary

```bash
# Build
wasm-pack build --target web

# Serve
python3 -m http.server 8000

# Use
1. Enter value
2. Click "Visualize"
3. See resonance across 71 layers
```

The tool is **self-modifying** - every input changes the entire display! 🎯
