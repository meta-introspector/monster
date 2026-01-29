# 👹 Monster Resonance Pipeline

**Date**: 2026-01-29  
**Location**: `pipeline/`  
**Status**: ✅ Complete

## Overview

**Pipeline**: Capture CPU registers → Apply harmonic analysis (FFT) → Find Monster group resonance

**Goal**: Discover which register patterns resonate most with Monster group primes!

## Architecture

```
Program Execution
       ↓
   [perf record]  ← Capture registers (AX, BX, CX, DX, SI, DI, R8-R15)
       ↓
   registers.json
       ↓
   [Julia FFT]    ← Apply harmonic analysis
       ↓
   harmonics.parquet
       ↓
   [Python]       ← Find Monster resonance
       ↓
   monster_resonance.parquet
```

## Components

### 1. Register Capture (`capture-registers`)

**What it does**:
- Runs program with `perf record`
- Captures 14 registers: AX, BX, CX, DX, SI, DI, R8, R9, R10, R11, R12, R13, R14, R15
- Parses to JSON

**Usage**:
```bash
capture-registers ./my_program registers.json
```

**Output**:
```json
[
  {
    "ip": "0x401234",
    "sym": "main",
    "regs": {
      "AX": 12345,
      "BX": 67890,
      ...
    }
  },
  ...
]
```

### 2. Harmonic Analysis (`analyze-harmonics`)

**What it does**:
- Loads register sequences
- Applies FFT to each register
- Computes power spectrum
- Finds dominant frequencies

**Usage**:
```bash
analyze-harmonics registers.json harmonics.parquet
```

**Algorithm**:
```julia
# For each register
values = [reg_value_1, reg_value_2, ...]

# Apply FFT
fft_result = fft(values)

# Power spectrum
power = abs2.(fft_result)

# Find top frequencies
top_freqs = sortperm(power, rev=true)[1:10]
```

**Output**: JSON with FFT results per register

### 3. Monster Resonance (`find-monster-resonance`)

**What it does**:
- Checks frequency divisibility by Monster primes
- Computes resonance score weighted by prime powers
- Ranks registers by Monster resonance

**Usage**:
```bash
find-monster-resonance harmonics.parquet monster.parquet
```

**Algorithm**:
```python
# Monster primes with powers
MONSTER_FACTORS = {
    2: 46, 3: 20, 5: 9, 7: 6, 11: 2, 13: 3,
    17: 1, 19: 1, 23: 1, 29: 1, 31: 1, 41: 1, 47: 1, 59: 1, 71: 1
}

# For each register's top frequencies
for prime, power in MONSTER_FACTORS.items():
    div_pct = (count divisible by prime) / total * 100
    resonance_score += div_pct * power

# Normalize
resonance_score /= sum(powers)
```

**Output**: Parquet with resonance scores

### 4. Full Pipeline (`monster-pipeline`)

**What it does**:
- Runs all 3 steps in sequence
- Generates all output files

**Usage**:
```bash
monster-pipeline ./my_program output_base
```

**Generates**:
- `output_base_registers.json`
- `output_base_harmonics.parquet`
- `output_base_monster.parquet`

## Installation

```bash
cd pipeline
nix develop
```

## Quick Start

### Run Full Pipeline

```bash
# Build a test program
cd ..
cargo build --release --bin main

# Run pipeline
cd pipeline
nix develop
monster-pipeline ../target/release/main test
```

### Step-by-Step

```bash
# Step 1: Capture
capture-registers ../target/release/main registers.json

# Step 2: Analyze
analyze-harmonics registers.json harmonics.parquet

# Step 3: Find resonance
find-monster-resonance harmonics.parquet monster.parquet
```

### View Results

```bash
python3 << EOF
import pandas as pd
df = pd.read_parquet('monster.parquet')
print(df.sort_values('resonance_score', ascending=False))
EOF
```

## Monster Primes

| Prime | Power | Weight |
|-------|-------|--------|
| 2 | 46 | Highest |
| 3 | 20 | High |
| 5 | 9 | Medium |
| 7 | 6 | Medium |
| 11 | 2 | Low |
| 13 | 3 | Low |
| 17-71 | 1 | Minimal |

**Total weight**: 138

## Expected Results

### High Resonance Registers

Registers with frequencies divisible by:
- **2** (46x weight) → 80%+ divisibility
- **3** (20x weight) → 50%+ divisibility
- **5** (9x weight) → 40%+ divisibility

**Example**:
```
Register: AX
Resonance Score: 85.3
  div_2: 90%
  div_3: 55%
  div_5: 42%
  div_7: 30%
```

### Low Resonance Registers

Random or non-Monster patterns:
```
Register: R15
Resonance Score: 12.7
  div_2: 51%
  div_3: 33%
  div_5: 20%
```

## Connection to Previous Work

### LLM Register Resonance (examples/ollama-monster/)

**Previous finding**:
- 80% of register values divisible by 2
- 49% divisible by 3
- 43% divisible by 5
- Same 5 primes [2,3,5,7,11] in 93.6% of error correction codes

**This pipeline**:
- Applies **harmonic analysis** (FFT) to register sequences
- Finds **frequency domain** resonance
- Weights by Monster prime powers

**Difference**:
- Previous: Time domain divisibility
- This: Frequency domain resonance

### Spherical Harmonics (harmonics_repos/)

**Connection**:
- FFT = Discrete Fourier Transform
- Spherical harmonics = Continuous Fourier on sphere
- Both decompose into orthogonal basis

**Analogy**:
```
Register sequence → FFT → Frequencies
Group function → Characters → Fourier coefficients
```

## Files

```
pipeline/
├── flake.nix ⭐⭐⭐
├── scripts/
│   ├── parse_registers.py ⭐⭐
│   ├── harmonic_analysis.jl ⭐⭐⭐
│   └── monster_resonance.py ⭐⭐⭐
└── PIPELINE.md (this file)
```

## Next Steps

### 1. Test with Monster Walk ⭐⭐⭐

```bash
cargo build --release --bin main
monster-pipeline ../target/release/main monster_walk
```

**Goal**: Find which registers resonate during Monster computation

### 2. Test with GAP ⭐⭐⭐

```bash
# Run GAP experiment
cd ../experiments
./01_monster_order.sh

# Capture registers
cd ../pipeline
monster-pipeline gap gap_monster
```

**Goal**: Find Monster resonance in GAP computation

### 3. Test with Julia Harmonics ⭐⭐⭐

```bash
# Run spherical harmonics
cd ../harmonics_repos
nix run .#test-spherical

# Capture registers
cd ../pipeline
monster-pipeline julia spherical_harmonics
```

**Goal**: Find resonance in harmonic computation itself!

### 4. Compare Across Programs ⭐⭐

```bash
# Run multiple programs
monster-pipeline prog1 test1
monster-pipeline prog2 test2
monster-pipeline prog3 test3

# Compare resonance
python3 << EOF
import pandas as pd
df1 = pd.read_parquet('test1_monster.parquet')
df2 = pd.read_parquet('test2_monster.parquet')
df3 = pd.read_parquet('test3_monster.parquet')
# Compare resonance scores
EOF
```

**Goal**: Find universal Monster patterns

### 5. Integrate with Lean4 ⭐⭐⭐

```lean
-- MonsterLean/RegisterResonance.lean
def registerResonanceScore (freqs : List ℕ) : ℚ :=
  -- Compute Monster resonance
  -- Prove properties
```

**Goal**: Formal verification of resonance

## Summary

✅ **4 pipeline tools** (capture, analyze, find, full)  
✅ **3 scripts** (Python parser, Julia FFT, Python resonance)  
✅ **Nix flake** with all dependencies  
✅ **Perf integration** for register capture  
✅ **FFT analysis** for frequency domain  
✅ **Monster resonance** weighted by prime powers  
✅ **Ready to test** with Monster Walk, GAP, Julia

**Pipeline complete, ready to find Monster resonance!** 👹✅
