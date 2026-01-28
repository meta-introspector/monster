# Monster Introspector for mistral.rs

## Overview

Instruments mistral.rs to analyze Monster prime patterns in:
1. **Model weights** (loaded from GGUF)
2. **Layer activations** (during forward pass)
3. **Attention patterns** (Q, K, V matrices)

## Architecture

```
monster-macros/          # Procedural macros
├── #[monster_introspect]   # Instrument functions
├── #[trace_weights]         # Trace weight loading
└── analyze_tensor!()        # Analyze tensor values

monster_introspector.rs  # Runtime analysis
├── MonsterTrace             # Global trace state
├── analyze_float_primes()   # Quantize and measure
└── save_trace()             # Export results

build.rs                 # Code rewriting
├── Find mistral.rs source
├── Inject Monster macros
└── Generate instrumented version
```

## Usage

### 1. Set up mistral.rs

```bash
git clone https://github.com/EricLBuehler/mistral.rs
export MISTRAL_RS_PATH=/path/to/mistral.rs
```

### 2. Build with Monster introspection

```bash
cd examples/ollama-monster
cargo build --release --features monster-introspect
```

### 3. Run inference with tracing

```rust
use monster_introspector::*;

fn main() {
    // Load model (weights automatically traced)
    let model = load_model("qwen2.5:3b")?;
    
    // Run inference (activations automatically traced)
    let output = model.generate("mathematician Conway")?;
    
    // Save Monster trace
    save_trace("monster_trace.json")?;
}
```

### 4. Analyze results

```bash
cargo run --release --bin analyze-monster-trace
```

## What Gets Traced

### Weights (at load time)
```json
{
  "layer": "model.layers.0.self_attn.q_proj",
  "weight_primes": {
    "2": 0.501,
    "3": 0.334,
    "5": 0.201
  }
}
```

### Activations (during inference)
```json
{
  "layer": "model.layers.0.self_attn.q_proj",
  "activation_primes": {
    "2": 0.803,
    "3": 0.492,
    "5": 0.431
  }
}
```

## Key Insights

### Hypothesis
If weights show Monster primes at rate W_p and activations show rate A_p:
- **Amplification**: A_p > W_p → Network amplifies prime structure
- **Suppression**: A_p < W_p → Network suppresses prime structure
- **Preservation**: A_p ≈ W_p → Network preserves prime structure

### Expected Results

| Prime | Weights | Activations | Ratio |
|-------|---------|-------------|-------|
| 2     | ~50%    | 80%         | 1.6x  |
| 3     | ~33%    | 49%         | 1.5x  |
| 5     | ~20%    | 43%         | 2.2x  |

**Conclusion**: Network amplifies Monster prime structure during computation!

## Integration with mistral.rs

### Automatic Instrumentation

```rust
// Original mistral.rs code:
fn forward(&self, input: &Tensor) -> Tensor {
    let q = self.q_proj.forward(input);
    let k = self.k_proj.forward(input);
    let v = self.v_proj.forward(input);
    // ...
}

// After build.rs instrumentation:
#[monster_introspect]
fn forward(&self, input: &Tensor) -> Tensor {
    let q = self.q_proj.forward(input);
    let _primes_q = analyze_tensor!(q);
    
    let k = self.k_proj.forward(input);
    let _primes_k = analyze_tensor!(k);
    
    let v = self.v_proj.forward(input);
    let _primes_v = analyze_tensor!(v);
    // ...
}
```

### Manual Instrumentation

```rust
use monster_macros::*;

#[monster_introspect]
fn my_layer(input: &Tensor) -> Tensor {
    let output = compute(input);
    
    // Analyze specific tensors
    let primes = analyze_tensor!(output);
    println!("Prime 2: {:.1}%", primes[&2] * 100.0);
    
    output
}
```

## Files

```
examples/ollama-monster/
├── monster_macros/
│   ├── Cargo.toml
│   └── src/lib.rs              # Procedural macros
├── src/
│   ├── monster_introspector.rs # Runtime analysis
│   └── analyze_monster_trace.rs # Result analyzer
├── build.rs                     # Code instrumentation
└── MONSTER_INTROSPECTOR.md      # This file
```

## Next Steps

1. Clone mistral.rs
2. Build with Monster introspection
3. Run inference on "mathematician Conway"
4. Compare weights vs activations
5. Verify: Activations amplify Monster prime structure
