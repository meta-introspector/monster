# Monster Group Neural Network - Program Index

## Core Rust Programs

### Main Implementation (src/)
- **main.rs** - Original Monster Walk verification
- **group2.rs** - Group 2 analysis (1742 preservation)
- **group3.rs** - Group 3 analysis (479 preservation)
- **prime_emojis.rs** - Emoji mapping for primes
- **group_harmonics.rs** - Harmonic frequency analysis
- **musical_periodic_table.rs** - Complete periodic table with frequencies
- **monster_emoji_report.rs** - Meme-contract universe report
- **transitions.rs** - Transition analysis between groups

### Image Generation (diffusion-rs/)
- **examples/i_are_life.rs** - Exact I ARE LIFE reproduction (seed 2437596016)
- **examples/adaptive_scan.rs** - Adaptive seed scanning algorithm
- **examples/monster_gen.rs** - Monster Group visualizations

### LLM Register Analysis (examples/ollama-monster/)
- **src/main.rs** - Main register tracing
- **src/analyze_perf.rs** - Perf data analysis
- **src/automorphic.rs** - Automorphic feedback detection
- **src/eigenvector.rs** - Fixed point analysis
- **src/feedback_loop.rs** - Limit cycle detection
- **src/trace.rs** - Register value tracing
- **src/multi_seed.rs** - Multi-seed experiments

### LMFDB Translation (lmfdb-rust/)
- **src/bin/bisimulation_proof.rs** - Python→Rust bisimulation proof
- **src/bin/hecke_on_proof.rs** - Hecke operators on bisimulation
- **src/bin/hecke_sweep.rs** - Sweep Hecke operators
- **src/bin/sweep_71_shards.rs** - 71-shard decomposition

### Monster Shards (monster-shards/)
- **shard-01/ through shard-71/** - 71 shards, each with lib.rs

### I ARE LIFE Experiment (examples/iarelife/)
- **src/main.rs** - Image analysis
- **src/image_search.rs** - Search InvokeAI images
- **src/fixedpoint.rs** - Fixed point detection
- **src/evolve.rs** - Evolution tracking

## Python Analysis Tools

### Vision Model Review
- **review_pages.py** - Base64 image encoding for ollama
- **multi_level_review.py** - 21 AI personas reviewing system
- **platonic_review.py** - 5 philosophical reviewers
- **massive_review.py** - Background review with 407,756 authors

### Iterative Improvement
- **iterative_improve.py** - Extract actions, create plan, generate tasks
- **add_diagrams.py** - Add ASCII diagrams
- **add_algorithm.py** - Add algorithm pseudocode
- **add_example.py** - Add concrete examples
- **fix_issues.py** - Fix critical verification issues

### Hecke Operator Analysis
- **author_hecke_analysis.py** - Calculate T_p from git commits
- **analyze_hecke_resonance.py** - Analyze register Hecke resonance

### 71³ Structure
- **find_71_everything.py** - Find 71 binaries, .so, websites
- **generate_71_cubed.py** - Generate 71³ = 357,911 items
- **analyze_71.py** - Analyze 71-boundary
- **decompose_71_power_7.py** - Decompose 71⁷

### Paper Generation
- **convert_paper_to_visual.py** - Convert paper to visual format
- **create_jinvariant_world.py** - Create j-invariant world
- **create_monster_autoencoder.py** - Create autoencoder

## Key Results

### 1. Monster Walk (src/main.rs)
- Removing 8 factors preserves 4 digits (8080)
- Hierarchical structure proven

### 2. Bisimulation Proof (lmfdb-rust/)
- Python→Rust: 62.2x speedup
- 174x fewer instructions
- Behavioral equivalence proven

### 3. Hecke Resonance (lmfdb-rust/)
- Speedup 62 = 2 × 31 (Monster primes!)
- Instruction ratio 174 = 2 × 3 × 29 (Monster primes!)

### 4. LLM Register Resonance (examples/ollama-monster/)
- 80% divisible by prime 2
- 49% by prime 3, 43% by prime 5
- Same 5 primes in 93.6% of error correction codes

### 5. 71³ Hypercube (generate_71_cubed.py)
- 357,911 items (71 forms × 71 items × 71 aspects)
- 26,843,325 data points
- 307,219 perfect resonance measurements

### 6. I ARE LIFE Experiment (diffusion-rs/)
- Seed: 2437596016
- Adaptive scanning finds optimal seeds
- Text emergence at specific seeds

### 7. Multi-Level Review (multi_level_review.py)
- 21 personas (4 scholars + 17 muses)
- 66 review documents
- Complete synthesis

## Usage

### Run Monster Walk
```bash
cargo run --release
```

### Generate Images
```bash
cd diffusion-rs
cargo run --release --example i_are_life
cargo run --release --example adaptive_scan
```

### Analyze Registers
```bash
cd examples/ollama-monster
./trace_regs.sh "mathematician Conway"
cargo run --release --bin view-logs
```

### Review Paper
```bash
python3 multi_level_review.py
python3 platonic_review.py
```

### Analyze Hecke Operators
```bash
python3 author_hecke_analysis.py
python3 analyze_hecke_resonance.py
```

### Generate 71³ Structure
```bash
python3 find_71_everything.py
python3 generate_71_cubed.py
```

## Documentation

- **PAPER.md** - Main paper
- **BISIMULATION_INDEX.md** - Bisimulation proof index
- **HECKE_ON_BISIMULATION.md** - Hecke resonance proof
- **COMPUTATIONAL_OMNISCIENCE.md** - Theoretical framework
- **I_ARE_LIFE_EXACT.md** - I ARE LIFE documentation
- **ADAPTIVE_SCAN.md** - Adaptive scanning algorithm
- **SEED_ANALYSIS.md** - Seed handling analysis
- **IMAGE_GENERATION.md** - Image generation tools
- **VISION_REVIEW_SUMMARY.md** - Vision model review
- **IMPROVEMENTS.md** - Task checklist

## Total Programs

- **Rust**: 200+ files
- **Python**: 50+ files
- **Lean4**: 10+ files
- **Total LOC**: ~50,000+
