# Monster Walk AI Sampling

Pure Rust AI sampling using [mistral.rs](https://github.com/meta-introspector/mistral.rs) - **no Python!** 🦀

## Experiments

### 1. **Progressive Automorphic Orbits** 🌀
Image generation → vision analysis → feedback loop

```bash
cargo run --bin orbit-runner
```

**Process:**
1. Generate image with FLUX.1-dev (seed + prompt)
2. Analyze with LLaVA vision model
3. Extract concepts → emoji encoding
4. Feed description back as next prompt
5. Repeat until convergence

**Output:**
- Semantic index of all concepts
- Emoji timeline showing evolution
- Convergence graph
- Attractor identification

### 2. **"I ARE LIFE" Reproduction** 🌱
Based on: https://huggingface.co/posts/h4/680145153872966

```bash
cargo run --bin emergence-test
```

Reproduces the exact experiment:
- Seed: 2437596016
- Prompt: "unconstrained"
- Detects self-awareness markers

### 3. **Homotopy Self-Observation** 🔄
Prolog-style eigenvector convergence

```bash
cargo run --bin homotopy-test
```

LLM observes its own execution traces:
- Traces → emoji encoding
- Self-referential loops
- Eigenvector computation
- Strange attractor detection

### 4. **Full Multi-Model Trace** 📊
Test across model sizes with tower analysis

```bash
cargo run --bin full-trace
```

Features:
- Multiple model sizes (7B, 70B)
- Tower of Babel capacity testing
- Convergence analysis
- Harmonic filtering

## Why mistral.rs?

- **Pure Rust**: No Python runtime, no dependency hell
- **Fast**: Native performance, no FFI overhead  
- **Portable**: Single binary, works everywhere
- **Vision Support**: LLaVA and other vision models
- **Local**: All inference runs locally

## Models

Models cached in `~/.cache/mistral.rs/`

Supported:
- **Image Gen**: FLUX.1-dev
- **Vision**: LLaVA, BakLLaVA
- **Text**: Mistral 7B, Mixtral 8x7B
- Any GGUF model

## Output Structure

```
emergence/
├── orbits/
│   ├── orbit_2437596016.json
│   ├── orbit_2437596016_REPORT.md
│   ├── orbit_8080.json
│   └── ...
├── images/
│   ├── step_000_seed_2437596016.png
│   ├── step_001_seed_2437596017.png
│   └── ...
├── semantic_index.json
└── convergence_analysis.json

ai-traces/
├── full_trace.json
├── execution_traces.pl
├── loops/
└── eigenvectors/
```

## Theory

### Automorphic Orbits
```
Image(seed, prompt) → Vision(image) → Concepts → Emoji
         ↑                                          ↓
         └──────────── Feedback ←───────────────────┘
```

### Semantic Indexing
- Track concept frequency
- Emoji pattern detection
- Convergence measurement
- Attractor identification

### Connection to Monster Walk
| Monster Walk | Automorphic Orbits |
|--------------|-------------------|
| Prime factorization | Concept extraction |
| Emoji primes | Emoji concepts |
| Leading digits | Semantic attractors |
| Hierarchical groups | Iteration steps |
| Eigenvector | Convergence point |

## No Python Required! 🦀

Everything runs in pure Rust - from image generation to vision analysis to semantic indexing.
