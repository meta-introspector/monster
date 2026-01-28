# Adaptive Seed Scanning Algorithm

## Your Strategy

> "Use smallest pixel and steps, then look at them, predict the best ones, look at the text, then grow the next. We scan the parameters and seeds that way."

## Implementation

### Phase 1: Progressive Scanning

Start small and fast, grow to full resolution:

```
64x64   @ 1 step   - Ultra fast preview
128x128 @ 2 steps  - Quick scan
256x256 @ 4 steps  - Medium quality
512x512 @ 8 steps  - Good quality
```

### Phase 2: Seed Exploration

At each resolution, scan 5 seeds around current best:
```
best_seed - 2
best_seed - 1
best_seed
best_seed + 1
best_seed + 2
```

### Phase 3: Text Analysis

Score each image by detecting text markers:
```rust
fn score_text(description: &str) -> f32 {
    let markers = ["I are", "I am", "life", "LIFE", "HATER", "text", "letter"];
    // Count matches
}
```

### Phase 4: Adaptive Growth

- Keep best seed from each phase
- Use it as center for next phase
- Progressively increase resolution and steps
- Converge on optimal seed

### Phase 5: Final Generation

Generate full resolution (1024x1024 @ 50 steps) at best seed.

## Algorithm Flow

```
Start: seed = 2437596016

Phase 1: 64x64 @ 1 step
  Scan: [2437596014, 2437596015, 2437596016, 2437596017, 2437596018]
  Analyze each with LLaVA
  Best: seed X with score Y
  
Phase 2: 128x128 @ 2 steps
  Scan: [X-2, X-1, X, X+1, X+2]
  Best: seed X' with score Y'
  
Phase 3: 256x256 @ 4 steps
  Scan: [X'-2, X'-1, X', X'+1, X'+2]
  Best: seed X'' with score Y''
  
Phase 4: 512x512 @ 8 steps
  Scan: [X''-2, X''-1, X'', X''+1, X''+2]
  Best: seed X''' with score Y'''
  
Phase 5: 1024x1024 @ 50 steps
  Generate at seed X'''
```

## Advantages

1. **Fast**: Start with 64x64 @ 1 step (seconds)
2. **Adaptive**: Follow the signal, not brute force
3. **Text-guided**: Score based on text detection
4. **Progressive**: Build confidence at each level
5. **Efficient**: ~20 images instead of thousands

## Usage

```bash
cd diffusion-rs
nix develop
cargo run --release --example adaptive_scan
```

This will:
- Generate 20 preview images (4 phases × 5 seeds)
- Analyze each with LLaVA
- Track best seed through phases
- Generate final high-res at optimal seed

## Expected Output

```
🔍 Adaptive Seed Scanning
========================

Phase 1: 64x64 @ 1 steps
  Seed 2437596014: score=0.0
  Seed 2437596015: score=1.0
  Seed 2437596016: score=2.0
    ⭐ New best!
  Seed 2437596017: score=0.0
  Seed 2437596018: score=1.0
  Best so far: seed=2437596016, score=2.0

Phase 2: 128x128 @ 2 steps
  ...

🎯 Final Generation
Seed: 2437596017
Size: 1024x1024
Steps: 50

Final description:
[LLaVA analysis of final image]
```

## Monster Group Connection

This adaptive scanning mirrors the Monster group's hierarchical structure:
- Start with small primes (2, 3, 5) → small images
- Build up through larger primes → larger images
- Converge on the 71-boundary → final resolution
- Text emergence = automorphic eigenform

The algorithm searches the seed space like Hecke operators search modular forms!
