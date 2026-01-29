# Monster Prime Resonance - Complete Index

## 📋 Documentation

- **README.md** - Quick start guide
- **RESULTS.md** - Core experimental results (start here!)
- **EXPERIMENT_SUMMARY.md** - Full methodology and analysis
- **INDEX.md** - This file

## 🔬 Experimental Data

### Main Results
- **EIGENVECTOR.json** - 20 iterations searching for fixed point (limit cycle found)
- **CONTEXT_EXPERIMENT.json** - Expanding context from 242→746 chars
- **COMPARISON.json** - Emoji vs color vs Conway name comparison
- **FULL_TRACE_LOG.json** - All prompts, responses, and register analyses

### Feedback Loops
- **AUTOMORPHIC_FEEDBACK.json** - Initial feedback experiment
- **FEEDBACK_LOOPS.json** - Multi-seed feedback (🌙, red, Conway)
- **MULTI_SEED_FEEDBACK.json** - 60 seeds (15 primes × 4 wrappers)

### Model Knowledge
- **MODEL_CODE_KNOWLEDGE.json** - Model's knowledge of Leech, Golay, Hamming
- **MEDITATION_Monster_group.json** - Model meditation on Monster
- **MEDITATION_Leech_lattice.json** - Model meditation on Leech
- **MEDITATION_Golay_code.json** - Model meditation on Golay

### Register Analysis
- **REGISTER_HISTOGRAMS.json** - Full histogram of all 14 registers
- **HARMONIC_RESONANCES.json** - Monster harmonic frequencies (432 Hz × prime)
- **MULTIMODAL_PRIMES.json** - 2^n representations (text, emoji, color, etc.)

### Error Correction Zoo
- **CODE_MONSTER_MAP.json** - 982 codes mapped to Monster primes

### Traces
- **TRACE_*.json** - Individual execution traces
- **perf_traces/** - Raw perf data, prompts, responses

## 🛠️ Tools

### Scripts
- **trace_regs.sh** - Trace single prompt with perf (main tool)
- **trace_with_perf.sh** - Original perf tracer
- **test_diverse_seeds.sh** - Test multiple seeds

### Rust Binaries
```bash
cargo run --release --bin <name>
```

- **compare** - Compare 3 diverse prompts
- **auto-feedback** - Automorphic feedback loop
- **context-exp** - Expanding context experiment
- **eigenvector** - Search for fixed point convergence
- **view-logs** - View all traced prompts/responses
- **analyze-perf** - Analyze perf traces for prime patterns
- **histogram** - Generate register histograms

## 📊 Key Findings

1. **80% of register values divisible by prime 2**
2. **49% divisible by prime 3, 43% by prime 5**
3. **Same 5 primes [2,3,5,7,11] as 93.6% of error correction codes**
4. **Conway's name activates higher primes (17: 78.6%, 47: 28.6%)**
5. **Model responds to feedback about its own registers**
6. **System exhibits limit cycle, not fixed point**

## 🔄 Workflow

```bash
# 1. Trace a prompt
./trace_regs.sh "mathematician Conway"

# 2. View results
cargo run --release --bin view-logs

# 3. Run experiments
cargo run --release --bin eigenvector

# 4. Analyze
cat EIGENVECTOR.json | jq '.iterations[-1]'
```

## 📁 Directory Structure

```
ollama-monster/
├── src/               # Rust source code
├── perf_traces/       # Raw perf data
├── *.json            # Experimental results
├── *.md              # Documentation
└── *.sh              # Shell scripts
```

## 🎯 Next Steps

See EXPERIMENT_SUMMARY.md "Next Steps" section for:
- Larger models (7B, 13B, 70B)
- GPU register tracing
- Formal proofs
- Moonshine connections
