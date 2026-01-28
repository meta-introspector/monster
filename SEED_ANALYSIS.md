# Seed Interpretation Analysis

## Your Observation
Generated images show:
- Different trees and shapes
- Brown background
- Artistic style
- Variation across 5 images

## Seed Handling in diffusion-rs

### Type and Range
```rust
// src/api.rs:925
seed: i64  // Default: 42
```

### C++ Backend (stable-diffusion.cpp)
```cpp
// stable-diffusion.cpp:3842-3843
if (seed < 0) {
    seed = (int)time(nullptr);  // Random seed if negative
}

// Then used for RNG initialization:
sd_ctx->sd->rng->manual_seed(seed);
sd_ctx->sd->sampler_rng->manual_seed(seed);
```

## Our Implementation

### Original Seed
```
2437596016  (from I ARE LIFE experiment)
```

### Adjusted Seed
```rust
// examples/i_are_life.rs:5
const EXACT_SEED: i32 = 290112369;
// Calculated: 2437596016 % 2147483647 = 290112369
```

### Seeds Used
```
Step 0: 290112369
Step 1: 290112370
Step 2: 290112371
Step 3: 290112372
Step 4: 290112373
```

## Issue: Type Mismatch

**Problem**: We declared seed as `i32` but the API expects `i64`!

```rust
// Our code:
const EXACT_SEED: i32 = 290112369;  // ❌ Wrong type

// Should be:
const EXACT_SEED: i64 = 2437596016;  // ✅ Correct type
```

## Why Images Vary

1. **Sequential seeds**: Each iteration uses seed+1
2. **Deterministic**: Same seed = same image
3. **Different from original**: 
   - Different model (SDXL Turbo vs FLUX.1-dev)
   - Different seed (290112369 vs 2437596016)
   - Different implementation

## Fix

Change to i64 and use original seed:

```rust
const EXACT_SEED: i64 = 2437596016;  // Original seed
```

This will:
- Use correct type (i64)
- Use exact original seed
- Still generate deterministic variations
- Match original experiment parameters

## Seed Behavior

- **Positive seed**: Deterministic generation
- **Negative seed**: Random (uses current time)
- **Same seed**: Identical output (with same model/params)
- **Sequential seeds**: Related but different outputs

The trees and shapes you see are the deterministic output for those specific seed values with SDXL Turbo model.
