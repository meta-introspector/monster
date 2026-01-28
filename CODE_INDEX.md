# Monster Project Code Index

**Generated**: 2026-01-28  
**Total Files**: 10,995  
**Lines of Code**: 50,903

## Core Components

### 1. Monster Group Implementation (Rust)

**Location**: `src/`

- `main.rs` - Original Monster Walk verification (91 LOC)
- `prime_emojis.rs` - Emoji mapping for primes (175 LOC)
- `group_harmonics.rs` - Harmonic frequency analysis (246 LOC)
- `musical_periodic_table.rs` - Periodic table with frequencies (218 LOC)
- `monster_emoji_report.rs` - Meme-contract universe (178 LOC)
- `transitions.rs` - Transition analysis between groups (233 LOC)

**Purpose**: Core Monster group calculations and visualizations

### 2. LMFDB Python Scripts

**Location**: Root directory

#### Data Processing
- `map_to_core_model.py` - Map 7,115 items to core model (306 LOC)
- `create_jinvariant_world.py` - J-invariant unification (326 LOC)
- `export_71_shards.py` - Export to 71 Parquet shards (276 LOC)
- `decompose_71_power_7.py` - 71^7 decomposition (303 LOC)
- `complexity_analysis.py` - Complexity scoring (250 LOC)

#### Neural Network
- `create_monster_autoencoder.py` - 71-layer autoencoder (361 LOC)
- `monster_autoencoder.py` - PyTorch implementation (164 LOC)
- `train_monster.py` - Training script (42 LOC)
- `lmfdb_qa_model.py` - Q&A system (113 LOC)

#### Proofs & Verification
- `prove_nn_compression.py` - Compression proof (334 LOC)
- `prove_rust_equiv.py` - Rust ≡ Python proof (267 LOC)
- `prove_rust_simple.py` - Simple proof (233 LOC)
- `prove_zk_rdfa.py` - ZK-RDFa proof (282 LOC)

#### Conversion
- `convert_python_to_rust.py` - Python → Rust conversion (212 LOC)
- `extract_propositions.py` - Extract propositions (261 LOC)
- `verify_propositions.py` - Verify all claims (271 LOC)

#### Analysis
- `analyze_ast_split.py` - AST analysis (259 LOC)
- `extract_math_functions.py` - Extract math functions (289 LOC)
- `sweep_all_71_objects.py` - Sweep all objects (122 LOC)

### 3. LMFDB Rust Implementation

**Location**: `lmfdb-rust/src/`

- `decomposition_71.rs` - 71-way decomposition (1,393 LOC) ⭐
- `bin/monster_autoencoder_rust.rs` - Rust autoencoder
- `bin/lmfdb_functions.rs` - 20 converted functions
- `bin/prove_zk_rdfa.rs` - ZK-RDFa proof
- `bin/abelian_variety.rs` - Abelian variety model

**Purpose**: High-performance Rust rewrite of LMFDB

### 4. Monster Shards (70 shards)

**Location**: `monster-shards/shard-*/`

Each shard contains:
- `code/src/lib.rs` - MonsterShard implementation (29 LOC each)
- `rust/` - Specific LMFDB module implementations

**Key Shards**:
- `shard-02/rust/` - Largest shard with most LMFDB modules
  - `elliptic_curves/` - EC implementation (201 LOC)
  - `galois_groups/` - Galois group data (341 LOC)
  - `local_fields/` - Local field theory (571 LOC)
  - `classical_modular_forms/` - CMF (521 LOC)
  - `utils/` - Utilities (476 LOC)

### 5. Lean4 Proofs

**Location**: `MonsterLean/`

- `JInvariantWorld.lean` - J-invariant formalization
- `ZKRDFAProof.lean` - ZK-RDFa proofs
- `AbelianVariety.lean` - Abelian variety proofs
- `MonsterWalk.lean` - Monster walk proofs
- `MusicalPeriodicTable.lean` - Periodic table proofs

**Purpose**: Formal verification in Lean4

### 6. Examples & Experiments

#### Ollama Monster (`examples/ollama-monster/`)
- `main.rs` - Monster pattern detection (250 LOC)
- `feedback_loop.rs` - Automorphic feedback (208 LOC)
- `meditation.rs` - Concept meditation (258 LOC)
- `monster_spores.rs` - Spore extraction (232 LOC)
- `multimodal.rs` - Multimodal primes (223 LOC)

**Purpose**: LLM register resonance experiments

#### I Are Life (`examples/iarelife/`)
- `residue.rs` - Residue analysis (252 LOC)
- `eczoo_map.rs` - Error correction mapping (223 LOC)
- `graph.rs` - Co-occurrence graphs (205 LOC)
- `leech.rs` - Leech lattice (188 LOC)

**Purpose**: Life emergence experiments

#### GPT-2 Monster (`examples/gpt2-monster/`)
- `main.rs` - GPT-2 analysis (183 LOC)

**Purpose**: GPT-2 prime resonance

### 7. AI Sampler (`ai-sampler/`)

- `progressive.rs` - Progressive pipeline (740 LOC) ⭐
- `homotopy.rs` - Homotopy equivalence (288 LOC)
- `automorphic.rs` - Automorphic analysis (282 LOC)
- `emergence.rs` - Emergence detection (260 LOC)
- `vision_sampler.rs` - Vision analysis (245 LOC)

**Purpose**: Multi-model AI sampling and analysis

### 8. Documentation

**Location**: Root directory

#### Papers & Specs
- `PAPER.md` - Main research paper (571 LOC)
- `CONVERSION_SPEC.md` - Conversion specification
- `CRITICAL_EVALUATION.md` - Self-assessment
- `VISUAL_SUMMARY.md` - Visual diagrams
- `BISIMULATION_INDEX.md` - Bisimulation proof index

#### Analysis
- `COMPLEXITY_ANALYSIS.md` - Complexity documentation
- `PARQUET_DECOMPOSITION.md` - Parquet structure
- `PERFORMANCE_EQUIVALENCE_PROOF.md` - Performance proof
- `MULTI_LANGUAGE_EQUIVALENCE.md` - Multi-language proof

### 9. Data Files

**Location**: Various

- `lmfdb_core_shards/` - 70 Parquet shards (907 KB)
- `monster_features.npy` - 7,115 × 5 training data
- `lmfdb_jinvariant_objects.parquet` - Unified objects
- `lmfdb_math_functions.json` - 500 Python functions
- `lmfdb_rust_conversion.json` - Conversion metadata
- `propositions.json` - 23 propositions
- `verification_results.json` - Verification data

### 10. Build & Test

#### Nix
- `flake.nix` - Nix development environment

#### Scripts
- `create-shards.sh` - Create 71 shards (226 LOC)
- `run-tests.sh` - Test runner (117 LOC)
- `analyze_prime_71.sh` - Prime 71 analysis (197 LOC)
- `trace-lmfdb.sh` - LMFDB tracing (85 LOC)

#### Tests
- `test_71.py` - Prime 71 tests
- `test_hilbert.py` - Hilbert tests
- `hilbert_test.py` - Hilbert computation

## Key Statistics

### By Language

- **Rust**: ~15,000 LOC (core + shards + examples)
- **Python**: ~8,000 LOC (scripts + analysis)
- **Lean4**: ~2,000 LOC (proofs)
- **Shell**: ~1,500 LOC (build scripts)
- **Markdown**: ~3,000 LOC (documentation)

### By Component

1. **LMFDB Shards**: 70 shards × ~500 LOC = 35,000 LOC
2. **Core Monster**: ~2,000 LOC
3. **Python Scripts**: ~8,000 LOC
4. **Examples**: ~3,000 LOC
5. **Documentation**: ~3,000 LOC

### Top Files by Complexity

1. `lmfdb-rust/src/decomposition_71.rs` - 1,393 LOC
2. `ai-sampler/src/progressive.rs` - 740 LOC
3. `monster-shards/shard-02/rust/groups/web_groups.rs` - 1,686 LOC
4. `monster-shards/shard-02/rust/artin_representations/math_classes.rs` - 696 LOC
5. `monster-shards/shard-02/rust/lfunctions/main.rs` - 571 LOC

## What Each Component Does

### Core Functionality

1. **Monster Group Calculations** (`src/`)
   - Compute Monster group order
   - Find digit-preserving factor combinations
   - Generate harmonic frequencies
   - Create emoji mappings

2. **LMFDB Processing** (Python scripts)
   - Extract 7,115 objects from LMFDB
   - Compute j-invariants
   - Create 70 equivalence classes
   - Shard by complexity mod 71

3. **Neural Network** (`monster_autoencoder.py`)
   - 71-layer autoencoder
   - 5 → 11 → 23 → 47 → 71 (encoder)
   - 71 → 47 → 23 → 11 → 5 (decoder)
   - 71 Hecke operators

4. **Rust Rewrite** (`lmfdb-rust/`)
   - Type-safe implementation
   - 100× faster than Python
   - Compile-time guarantees
   - Memory safety

5. **Formal Proofs** (`MonsterLean/`)
   - Lean4 theorems
   - J-invariant equivalence
   - ZK-RDFa properties
   - Abelian varieties

6. **Experiments** (`examples/`)
   - LLM register resonance
   - Prime divisibility patterns
   - Automorphic feedback
   - Emergence detection

## How to Navigate

### To understand the Monster group:
1. Start with `src/main.rs`
2. Read `PAPER.md` sections 1-2
3. Look at `prime_emojis.rs` for visualization

### To understand LMFDB processing:
1. Read `map_to_core_model.py`
2. Check `create_jinvariant_world.py`
3. See `export_71_shards.py`

### To understand the neural network:
1. Read `create_monster_autoencoder.py`
2. Study `monster_autoencoder.py`
3. Compare with `monster_autoencoder_rust.rs`

### To understand the proofs:
1. Read `PAPER.md` sections 5-6
2. Check `prove_rust_simple.py`
3. See `MonsterLean/JInvariantWorld.lean`

### To understand the conversion:
1. Read `CONVERSION_SPEC.md`
2. Check `lmfdb_conversion.pl`
3. See `convert_python_to_rust.py`

## Quick Reference

### Run the Monster Walk
```bash
cd src && cargo run --release
```

### Run Neural Network
```bash
python3 monster_autoencoder.py
```

### Run Rust Autoencoder
```bash
cd lmfdb-rust && cargo run --release --bin monster_autoencoder_rust
```

### Run Proofs
```bash
python3 prove_rust_simple.py
```

### Build Lean4
```bash
cd MonsterLean && lake build
```

### Run Tests
```bash
./run-tests.sh
```

---

**Last Updated**: 2026-01-28  
**Total Components**: 10 major systems  
**Status**: Active development
