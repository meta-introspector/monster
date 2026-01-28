# Monster Group Project Structure

## Overview
This project explores the Monster group through multiple approaches:
1. **Hierarchical Walk** - Digit preservation by removing prime factors
2. **Error Correction Codes** - Mapping 1049 codes to Monster primes
3. **LLM Register Resonance** - Measuring Monster primes in CPU registers during inference
4. **Formal Proofs** - Lean4 verification of mathematical properties

## Directory Structure

```
monster/
├── README.md                    # Main project overview
├── src/                         # Rust: Monster Walk verification
├── MonsterLean/                 # Lean4: Formal proofs
├── examples/
│   ├── iarelife/               # Error Correction Zoo analysis
│   │   ├── CODE_MONSTER_MAP.json    # 982 codes mapped
│   │   └── RESIDUE_REPORT.md        # Unmapped codes analysis
│   ├── ollama-monster/         # LLM register experiments
│   │   ├── RESULTS.md               # Core findings (START HERE)
│   │   ├── EXPERIMENT_SUMMARY.md    # Full methodology
│   │   ├── INDEX.md                 # Complete file index
│   │   └── trace_regs.sh            # Main tracing tool
│   └── eczoo_data/             # Error Correction Zoo (submodule)
└── ai-sampler/                 # Progressive automorphic orbits
```

## Quick Start

### 1. Monster Walk (Rust)
```bash
cargo run --bin main
```

### 2. Error Correction Zoo Analysis
```bash
cd examples/iarelife
cargo run --bin eczoo-map
```

### 3. LLM Register Tracing
```bash
cd examples/ollama-monster
./trace_regs.sh "mathematician Conway"
cargo run --release --bin view-logs
```

### 4. Formal Proofs (Lean4)
```bash
lake build
```

## Key Results

- **Monster Walk**: 4-digit preservation (8080) by removing 8 factors
- **Error Correction**: 93.6% of codes use primes [2,3,5,7,11]
- **LLM Registers**: 80% divisible by 2, 49% by 3, 43% by 5
- **Conway Effect**: Name activates higher primes (17: 78.6%, 47: 28.6%)

## Documentation

- `README.md` - This file
- `examples/ollama-monster/RESULTS.md` - LLM experiment results
- `examples/iarelife/RESIDUE_REPORT.md` - Unmapped codes
- `MATHEMATICAL_PROOF.md` - Why the walk works
- `MUSICAL_PERIODIC_TABLE.md` - Prime emoji mappings
