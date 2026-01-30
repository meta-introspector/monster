# WASM Lattice Matcher

Interactive tool for matching Value Lattice with Qwen Shards using prime resonance.

## Features

✅ **Real-time Matching**: Enter any value, see which layers resonate  
✅ **71-Layer Visualization**: Spiral display of all Qwen layers  
✅ **Prime Resonance**: Check divisibility by Monster primes  
✅ **Self-Modifying Display**: Canvas updates based on resonance field  
✅ **Interactive Controls**: Slider to navigate layers, scan all at once  

## Architecture

```
Value Input → WASM Matcher → Prime Resonance → Visual Display
     ↓              ↓                ↓               ↓
   "24"      Gödel: 155        2,3,5,7,11...    Spiral + Bars
```

## Build

```bash
# Install wasm-pack
cargo install wasm-pack

# Build
cd wasm_lattice_matcher
wasm-pack build --target web

# Serve
python3 -m http.server 8000
# Open http://localhost:8000
```

## Usage

1. **Enter Value**: Type any number (e.g., 24, 71)
2. **Match**: See which layers resonate with that value
3. **Scan All**: Check resonance across all 71 layers
4. **Visualize**: See resonance field as bars + 71-layer spiral

## Resonance Algorithm

```rust
fn match_with_prime(value: u32, prime: u32) -> f32 {
    if value % prime == 0 {
        1.0  // Perfect resonance
    } else {
        1.0 / ((value % prime) as f32 + 1.0)  // Partial resonance
    }
}
```

## Display Modes

### Bar Chart
- X-axis: 15 Monster primes
- Y-axis: Resonance strength (0-1)
- Color: Green (high) → Red (low)

### Spiral
- 71 layers in 5-turn spiral
- Yellow dots: Resonant layers (>0.5)
- Green line: Layer connections

## Examples

**Value 24**:
- Layer 0 (Prime 2): 1.0 ✅
- Layer 1 (Prime 3): 1.0 ✅
- Layer 4 (Prime 11): 0.08
- Layer 14 (Prime 71): 0.04

**Value 71**:
- Layer 14 (Prime 71): 1.0 ✅
- Layer 0 (Prime 2): 0.5
- Layer 1 (Prime 3): 0.33

## Deploy

### GitHub Pages
```bash
git add wasm_lattice_matcher/
git commit -m "Add WASM lattice matcher"
git push
# Enable Pages → /wasm_lattice_matcher
```

### Cloudflare Pages
```bash
wrangler pages publish wasm_lattice_matcher/ \
  --project-name=monster-lattice-matcher
```

## API

```javascript
import init, { LatticeMatcher } from './pkg/wasm_lattice_matcher.js';

await init();
const matcher = new LatticeMatcher();

// Add values
matcher.add_value("24", 155);
matcher.add_value("71", 21);

// Match
const matches = matcher.find_matches("24");
console.log(matches);

// Compute resonance field
const field = matcher.compute_resonance_field("71");
// Returns: [f32; 15] - resonance with each prime
```

## Self-Modifying Display

The display updates in real-time based on:
1. **Input value** changes resonance field
2. **Layer slider** changes active prime
3. **Scan** highlights all resonant layers
4. **Visualize** redraws spiral with new resonances

The canvas is **self-modifying** - it recalculates and redraws based on the current state of the value lattice and Qwen shards! 🎯
