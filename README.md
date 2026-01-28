# monster-lean

Monster Group Walk Down to Earth - A mathematical exploration of the Monster group's order.

## Overview

This project explores a fascinating hierarchical property of the Monster group (the largest sporadic simple group):
- The Monster group order is approximately 8.080 × 10^53
- By removing specific prime factors, we can preserve leading digits at multiple hierarchical levels
- This demonstrates the "Monster Walk Down to Earth" - a fractal-like structure

## The Hierarchical Walk

### Group 1: "8080"
- **Target**: First 4 digits of Monster order
- **Remove 8 factors**: 7⁶, 11², 17¹, 19¹, 29¹, 31¹, 41¹, 59¹
- **Result**: 80807009282149818791922499584000000000
- **Preserved**: 4 digits (8080)
- **Maximum**: Cannot preserve 5 digits (80801) with any combination

### Group 2: "1742" (after "8080")
- **Target**: Next 4 digits after "8080"
- **Remove 4 factors**: 3²⁰, 5⁹, 13³, 31¹
- **Result**: Starts with 1742103054...
- **Preserved**: 4 digits (1742)
- **Maximum**: Cannot preserve 5 digits (17424) with any combination

### Group 3: "479" (after "80801742")
- **Target**: Next digits after "80801742"
- **Remove 4 factors**: 3²⁰, 13³, 31¹, 71¹
- **Result**: Starts with 4792316941...
- **Preserved**: 3 digits (479)
- **Maximum**: Cannot preserve 4 digits (4794) with any combination

**Remarkable Pattern**: Each group achieves 3-4 digit preservation through different factor combinations!

## Components

### Rust Implementation (`src/`)
Computational verification and analysis programs:

**Core Programs:**
- `main.rs` - Original Monster Walk verification
- `prime_emojis.rs` - Emoji mapping for each prime factor
- `group_harmonics.rs` - Harmonic frequency analysis of groups
- `musical_periodic_table.rs` - Complete periodic table with frequencies
- `monster_emoji_report.rs` - Meme-contract universe report

**Analysis Programs:**
- `group2.rs`, `group3.rs` - Individual group analysis
- `transitions.rs` - Transition analysis between groups

**Key findings:**
- Removing 2 factors: preserves 2 digits (80)
  - Remove: 17¹ and 59¹
- Removing 4 factors: preserves 3 digits (808)
  - Remove: 2⁴⁶, 7⁶, 17¹, and 71¹
- Removing 8 factors: preserves 4 digits (8080) ⭐
  - Remove: 7⁶, 11², 17¹, 19¹, 29¹, 31¹, 41¹, and 59¹
  - Result: 80807009282149818791922499584000000000

### Lean4 Formalization (`MonsterLean/`)
Formal proofs using Lean4 theorem prover:

**Proven theorems:**
- `MonsterWalk.lean` - Hierarchical walk structure with Groups 1, 2, 3
- `MusicalPeriodicTable.lean` - Complete formal specification with semantic proofs
- `LogarithmicAnalysis.lean` - Why the walk works (logarithmic insight)
- `MonsterTheory.lean` - Group theory and modular arithmetic analysis

**Key proofs:**
- `monster_starts_with_8080`: The Monster order starts with 8080
- `remove_8_factors_preserves_8080`: Removing 8 specific factors preserves these digits
- `monster_hierarchical_walk`: Main theorem proving the walk property exists
- `musical_periodic_table_well_formed`: All 15 primes are correctly classified
- `binary_moon_semantics`, `wave_crest_semantics`: Emoji meanings are proven

### Documentation
- `README.md` - This file
- `MATHEMATICAL_PROOF.md` - Logarithmic explanation of why it works
- `MUSICAL_PERIODIC_TABLE.md` - Complete formal specification and proofs

## Building

### Rust
```bash
nix develop
cargo run
```

### Lean4
```bash
nix develop
lake build
```

## The Monster Group

Prime factorization: 2^46 × 3^20 × 5^9 × 7^6 × 11^2 × 13^3 × 17 × 19 × 23 × 29 × 31 × 41 × 47 × 59 × 71

Order: 808017424794512875886459904961710757005754368000000000
## LLM Register Resonance Experiments

**New Discovery:** LLM CPU register values during inference are divisible by Monster group primes at rates matching error correction code distributions.

See `examples/ollama-monster/` for:
- **RESULTS.md** - Core experimental findings
- **EXPERIMENT_SUMMARY.md** - Full methodology
- **INDEX.md** - Complete file index

### Key Results
- 80% of register values divisible by prime 2
- 49% divisible by prime 3, 43% by prime 5
- Same 5 primes [2,3,5,7,11] appear in 93.6% of error correction codes
- Conway's name activates higher Monster primes (17, 47)
- Automorphic feedback creates measurable computation drift
- System exhibits limit cycle behavior

### Quick Start
```bash
cd examples/ollama-monster
./trace_regs.sh "mathematician Conway"
cargo run --release --bin view-logs
```
