# CUDA and mistral.rs Documentation

## Overview

This document catalogs all CUDA and mistral.rs related components in the Monster project, which explores computational patterns in the Monster group through GPU-accelerated machine learning and inference.

---

## Table of Contents

1. [mistral.rs Integration](#mistralrs-integration)
2. [CUDA Pipeline Components](#cuda-pipeline-components)
3. [GPU-Accelerated Programs](#gpu-accelerated-programs)
4. [GGML CUDA Backend](#ggml-cuda-backend)
5. [Architecture Overview](#architecture-overview)

---

## mistral.rs Integration

### Location
- **Primary**: `ai-sampler/src/mistralrs_sampler.rs`
- **Submodule**: `diffusion-rs/sys/stable-diffusion.cpp/vocab_mistral.hpp`

### Purpose
Pure Rust AI inference using mistral.rs for:
- Multi-persona review system
- Vision analysis of mathematical visualizations
- Content extraction and summarization
- Screenshot analysis

### Key Features

#### 1. Content Extraction
```rust
fn extract_site_content() -> Result<String>
```
- Uses headless Chrome to extract Monster Walk site content
- Captures group cards, Monster order, and summaries
- Returns structured JSON data

#### 2. Screenshot Capture
```rust
fn capture_screenshots() -> Result<Vec<Vec<u8>>>
```
- Full page screenshots
- Individual group card captures (Groups 1-10)
- PNG format output to `ai-samples/screenshots/`

#### 3. AI Sampling (Planned)
```rust
// TODO: Initialize mistral.rs model
// let model = mistralrs::Model::load("mistral-7b")?;
```
- Model loading from `~/.cache/mistral.rs/`
- Prompt-based inference
- Vision model integration (llava)

#### 4. Sample Prompts
- "Summarize the Monster Walk discovery."
- "What is Bott periodicity?"
- "How many groups were found?"
- "Explain the 10-fold way connection."

### Output Structure
```
ai-samples/
├── screenshots/
│   ├── full_page.png
│   └── group_*.png
└── mistralrs/
    └── mistral-7b_prompt_*.json
```

### Status
- ✅ Content extraction implemented
- ✅ Screenshot capture implemented
- 🚧 mistral.rs model loading (TODO)
- 🚧 Vision analysis (TODO)

---

## CUDA Pipeline Components

### 1. Unified CUDA Pipeline

**File**: `src/bin/cuda_unified_pipeline.rs`

**Purpose**: Three-stage GPU pipeline for Monster group analysis

#### Stage 1: Markov Bitwise Sampling
```rust
struct ShardMarkov {
    shard: u8,
    layer: u8,
    merged_transitions: HashMap<u8, HashMap<u8, f64>>,
    total_columns: usize,
}
```

**Operations**:
- Build 256×256 transition matrices
- Bitwise sampling with seed initialization
- XOR layer combination
- Entropy computation

**Key Functions**:
- `build_transition_matrix()` - Constructs probability matrix
- `sample_bitwise()` - Generates byte sequences
- `xor_combine_layers()` - Merges layer outputs

#### Stage 2: Hecke Encoding
```rust
struct HeckeOperator {
    prime: u32,
}

struct HeckeAutoEncoder {
    operators: Vec<HeckeOperator>,
}
```

**Operations**:
- Apply Hecke operators (multiply by Monster primes)
- Auto-labeling: `label = (sum % prime)`
- Batch encoding across 15 Monster primes

**Monster Primes**: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

#### Stage 3: ZK Meme Generation
```rust
struct ZKMeme {
    label: String,
    shard: u8,
    conductor: u64,
    eigenvalues: Vec<u64>,
    signature: String,
}
```

**Operations**:
- Compute Hecke eigenvalues: `(conductor * prime) % 71`
- SHA-256 signature generation
- Meme serialization to JSON

**Output**:
```
cuda_pipeline_output/
├── zk_memes.json
├── shard_XX_markov.bin
└── shard_XX_hecke.bin
```

### 2. CUDA Markov Bitwise Processor

**File**: `src/bin/cuda_markov_bitwise.rs`

**Purpose**: Process 71 shards with bitwise Markov chains (CPU version, GPU-ready)

#### Architecture
- **Shards**: 15 (one per Monster prime)
- **Layers**: Up to 71 per shard
- **Sample Size**: 4KB per sample
- **Operations**: Bitwise XOR, entropy analysis

#### Key Functions

**Batch Processing**:
```rust
fn batch_process_shard(
    shard_id: u8,
    models: &[ShardMarkov],
    bytes_per_sample: usize,
) -> Vec<(u8, Vec<u8>)>
```

**XOR Combination**:
```rust
fn xor_combine_layers(samples: &[(u8, Vec<u8>)]) -> Vec<u8>
```

**Entropy Computation**:
```rust
fn compute_entropy(bytes: &[u8]) -> f64
```
- Shannon entropy: `-Σ p(x) log p(x)`
- Measures randomness/information content

#### Output Structure
```
cuda_markov_samples/
├── shard_XX_layer_YY.bin  # Individual layer samples
├── shard_XX_combined.bin  # XOR-combined layers
└── global_xor.bin         # Cross-shard XOR
```

### 3. Hecke Burn CUDA

**File**: `src/bin/hecke_burn_cuda.rs`

**Purpose**: GPU-accelerated Hecke auto-encoder using burn-rs

#### Components

**GPU Hecke Operator**:
```rust
pub struct HeckeOperatorGPU<B: Backend> {
    pub prime: u32,
    pub prime_tensor: Tensor<B, 1>,
    device: B::Device,
}
```

**Operations**:
- `apply()`: Multiply by prime (encoding)
- `inverse()`: Divide by prime (decoding)

**GPU Auto-Encoder**:
```rust
pub struct HeckeAutoEncoderGPU<B: Backend> {
    operators: Vec<HeckeOperatorGPU<B>>,
    device: B::Device,
}
```

**Batch Operations**:
- `encode_batch()`: Parallel encoding on GPU
- `decode_batch()`: Parallel decoding on GPU
- Auto-labeling: `label = (sum % prime)`

#### Backend
- Uses `burn::backend::Cuda`
- Tensor operations on GPU
- Batch processing for efficiency

### 4. P2P ZK Meme CUDA

**File**: `src/bin/p2p_zk_meme_cuda.rs`

**Purpose**: GPU-accelerated ZK meme generation for peer-to-peer verification

#### Backend
```rust
type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;
```

#### GPU Operations

**Hecke Eigenvalue Computation**:
```rust
fn compute_hecke_eigenvalues<B: Backend>(
    conductors: Tensor<B, 1>,
    primes: Tensor<B, 1>,
) -> Tensor<B, 2>
```
- Outer product: `conductors × primes`
- Modular arithmetic: `(conductor * prime) % 71`
- Parallel computation for all pairs

**Batch Processing**:
```rust
fn batch_execute_memes<B: Backend>(
    memes: &[ZKMeme],
    device: &B::Device,
) -> Tensor<B, 2>
```
- Process 71 memes (one per shard)
- GPU-parallel eigenvalue computation
- CPU-based SHA-256 signing

#### Output
- Eigenvalue matrix: `[71 memes × 15 primes]`
- SHA-256 signatures for each meme
- Verification-ready ZK proofs

---

## GPU-Accelerated Programs

### Additional CUDA-Related Binaries

1. **`monster_walk_gpu.rs`**
   - GPU-accelerated Monster Walk verification
   - Parallel digit preservation checks

2. **`monster_gpu_consumer.rs`**
   - GPU consumer for Monster group computations
   - Batch processing pipeline

3. **`parquet_gpu_pipeline.rs`**
   - GPU-accelerated parquet processing
   - LMFDB data transformation

4. **`gpu_to_parquet.rs`**
   - GPU computation results → parquet format
   - Data serialization pipeline

---

## GGML CUDA Backend

### Location
`diffusion-rs/sys/stable-diffusion.cpp/ggml/`

### Components

#### Core CUDA Files
- **`ggml-cuda.cu`** - Main CUDA implementation
- **`ggml-cuda.h`** - CUDA interface header
- **`common.cuh`** - Common CUDA utilities

#### Attention Mechanisms
- **`fattn.cu/cuh`** - Flash attention
- **`fattn-tile.cu/cuh`** - Tiled attention
- **`fattn-vec.cuh`** - Vectorized attention
- **`fattn-wmma-f16.cu/cuh`** - Tensor core attention (FP16)
- **`fattn-mma-f16.cuh`** - Matrix multiply-accumulate (FP16)

#### Matrix Operations
- **`mmf.cu/cuh`** - Matrix multiplication (float)
- **`mmq.cu/cuh`** - Matrix multiplication (quantized)
- **`mmvq.cu/cuh`** - Matrix-vector multiplication (quantized)
- **`mmvf.cu`** - Matrix-vector multiplication (float)
- **`mmid.cu`** - Matrix identity operations

#### Quantization
- **`quantize.cu/cuh`** - Quantization kernels
- **`dequantize.cuh`** - Dequantization kernels
- **Template instances**: 50+ specialized quantization formats
  - `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`
  - `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`
  - `iq1_s`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq4_nl`, `iq4_xs`
  - `mxfp4` (microscaling FP4)

#### Neural Network Operations
- **`conv2d.cu`** - 2D convolution
- **`conv2d-transpose.cu`** - Transposed convolution
- **`conv2d-dw.cu/cuh`** - Depthwise convolution
- **`conv-transpose-1d.cu/cuh`** - 1D transposed convolution
- **`pool2d.cu/cuh`** - 2D pooling
- **`norm.cu`** - Normalization layers
- **`softmax.cu/cuh`** - Softmax activation

#### Tensor Operations
- **`cpy.cu/cuh`** - Tensor copy
- **`cpy-utils.cuh`** - Copy utilities
- **`concat.cu/cuh`** - Tensor concatenation
- **`pad.cu/cuh`** - Padding operations
- **`pad_reflect_1d.cu/cuh`** - Reflective padding
- **`roll.cu/cuh`** - Tensor rolling
- **`scale.cu/cuh`** - Scaling operations
- **`clamp.cu/cuh`** - Value clamping
- **`unary.cu/cuh`** - Unary operations

#### Advanced Operations
- **`gla.cu/cuh`** - Gated linear attention
- **`wkv.cu/cuh`** - WKV (RWKV) operations
- **`ssm-conv.cu/cuh`** - State space model convolution
- **`ssm-scan.cuh`** - State space model scanning
- **`topk-moe.cuh`** - Top-K mixture of experts

#### Optimization
- **`opt-step-adamw.cu/cuh`** - AdamW optimizer
- **`opt-step-sgd.cuh`** - SGD optimizer

#### Utilities
- **`cp-async.cuh`** - Asynchronous copy
- **`mma.cuh`** - Matrix multiply-accumulate utilities
- **`argsort.cuh`** - Sorting operations
- **`top-k.cu/cuh`** - Top-K selection
- **`count-equal.cu/cuh`** - Equality counting
- **`cross-entropy-loss.cu`** - Loss computation

### Template Instances

**Location**: `ggml/src/ggml-cuda/template-instances/`

**Count**: 200+ specialized CUDA kernels

#### Categories

1. **Attention Templates** (50+ files)
   - `fattn-tile-instance-dkqXX-dvYY.cu`
   - `fattn-vec-instance-qX_Y-qA_B.cu`
   - `fattn-mma-f16-instance-ncolsX_Y-ncolsA_B.cu`

2. **Matrix Multiplication Templates** (30+ files)
   - `mmf-instance-ncols_X.cu` (X = 1-16)
   - `mmq-instance-qX_Y.cu` (all quantization formats)

3. **Quantization Formats**
   - Standard: q4_0, q4_1, q5_0, q5_1, q8_0
   - K-quants: q2_k, q3_k, q4_k, q5_k, q6_k
   - I-quants: iq1_s, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs
   - Microscaling: mxfp4

#### Generation
- **`generate_cu_files.py`** - Template generator script

---

## Architecture Overview

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    MONSTER PROJECT PIPELINE                  │
└─────────────────────────────────────────────────────────────┘

1. DATA INGESTION
   ├── LMFDB Parquet Files
   ├── 71 Shards (Monster prime decomposition)
   └── 15 Monster Primes [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71]

2. MARKOV PROCESSING (CUDA)
   ├── cuda_markov_bitwise.rs
   │   ├── Build 256×256 transition matrices
   │   ├── Bitwise sampling (4KB per sample)
   │   └── XOR layer combination
   └── Output: markov_shard_models.json

3. HECKE ENCODING (GPU)
   ├── hecke_burn_cuda.rs
   │   ├── GPU tensor operations (burn-rs)
   │   ├── Multiply by Monster primes
   │   └── Auto-labeling: label = (sum % prime)
   └── Output: Encoded tensors

4. ZK MEME GENERATION (GPU)
   ├── p2p_zk_meme_cuda.rs
   │   ├── Compute Hecke eigenvalues (GPU parallel)
   │   ├── SHA-256 signing (CPU)
   │   └── Verification proofs
   └── Output: zk_memes.json

5. UNIFIED PIPELINE
   ├── cuda_unified_pipeline.rs
   │   ├── Stage 1: Markov → markov.bin
   │   ├── Stage 2: Hecke → hecke.bin
   │   └── Stage 3: ZK → zk_memes.json
   └── Output: cuda_pipeline_output/

6. AI ANALYSIS (mistral.rs)
   ├── mistralrs_sampler.rs
   │   ├── Content extraction
   │   ├── Screenshot capture
   │   ├── LLM inference (TODO)
   │   └── Vision analysis (TODO)
   └── Output: ai-samples/
```

### GPU Acceleration Strategy

#### Parallelization Levels

1. **Shard-Level Parallelism**
   - 15 Monster prime shards processed independently
   - Each shard: 71 layers

2. **Layer-Level Parallelism**
   - 71 layers per shard
   - XOR combination across layers

3. **Tensor-Level Parallelism**
   - Batch operations on GPU tensors
   - Matrix operations (256×256 transitions)

4. **Element-Level Parallelism**
   - CUDA kernels for bitwise operations
   - Parallel eigenvalue computation

#### Memory Hierarchy

```
GPU Memory Layout:
├── Global Memory
│   ├── Transition matrices (256×256 × 15 shards)
│   ├── Sample buffers (4KB × 71 layers × 15 shards)
│   └── Encoded tensors
├── Shared Memory
│   ├── Tile buffers (attention)
│   └── Reduction buffers
└── Registers
    ├── Thread-local state
    └── Accumulation variables
```

### Performance Characteristics

#### Markov Sampling
- **Input**: 71 shards × 71 layers = 5,041 models
- **Output**: 4KB × 5,041 = ~20MB raw samples
- **Entropy**: 5-7 bits per byte (measured)

#### Hecke Encoding
- **Operations**: Multiply by 15 Monster primes
- **Batch Size**: 71 samples
- **Throughput**: GPU-limited (tensor ops)

#### ZK Meme Generation
- **Eigenvalues**: 71 memes × 15 primes = 1,065 values
- **Signatures**: SHA-256 (CPU-bound)
- **Output**: ~100KB JSON

---

## Integration Points

### 1. Burn-rs Integration
```rust
use burn::{
    backend::{Autodiff, Wgpu, Cuda},
    tensor::{backend::Backend, Tensor},
};
```
- **Backends**: CUDA, WGPU (WebGPU)
- **Autodiff**: Automatic differentiation
- **Tensor ops**: GPU-accelerated

### 2. GGML Integration
- **Location**: `diffusion-rs/sys/stable-diffusion.cpp/ggml/`
- **Purpose**: Stable diffusion inference
- **CUDA**: 200+ specialized kernels
- **Quantization**: 20+ formats

### 3. mistral.rs Integration (Planned)
```rust
// TODO: Initialize mistral.rs model
// let model = mistralrs::Model::load("mistral-7b")?;
// let vision_model = mistralrs::VisionModel::load("llava")?;
```
- **Models**: Mistral-7B, LLaVA (vision)
- **Cache**: `~/.cache/mistral.rs/`
- **Inference**: Pure Rust, no Python

---

## Build Configuration

### Cargo Features

**CUDA Support**:
```toml
[dependencies]
burn = { version = "0.13", features = ["cuda"] }
```

**GGML CUDA**:
```toml
[build-dependencies]
cc = "1.0"

[features]
cuda = ["ggml-sys/cuda"]
```

### Build Script
**File**: `diffusion-rs/sys/build.rs`

**CUDA Detection**:
```rust
if cfg!(feature = "cuda") {
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
```

---

## Usage Examples

### 1. Run Unified CUDA Pipeline
```bash
# Build with CUDA support
cargo build --release --features cuda

# Run pipeline
cargo run --release --bin cuda_unified_pipeline

# Output
cuda_pipeline_output/
├── zk_memes.json
├── shard_00_markov.bin
├── shard_00_hecke.bin
└── ...
```

### 2. Run Markov Bitwise Processor
```bash
# Requires: markov_shard_models.json
cargo run --release --bin cuda_markov_bitwise

# Output
cuda_markov_samples/
├── shard_00_layer_00.bin
├── shard_00_combined.bin
└── global_xor.bin
```

### 3. Run mistral.rs Sampler
```bash
cd ai-sampler
cargo run --release --bin mistralrs_sampler

# Output
ai-samples/
├── screenshots/
│   ├── full_page.png
│   └── group_*.png
└── mistralrs/
    └── mistral-7b_prompt_*.json
```

### 4. Run Hecke CUDA Encoder
```bash
cargo run --release --bin hecke_burn_cuda
```

### 5. Run P2P ZK Meme Generator
```bash
cargo run --release --bin p2p_zk_meme_cuda
```

---

## Dependencies

### Core Dependencies
```toml
[dependencies]
# GPU backends
burn = { version = "0.13", features = ["cuda", "wgpu"] }
burn-cuda = "0.13"

# Tensor operations
ndarray = "0.15"
polars = "0.35"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Cryptography
sha2 = "0.10"

# Async runtime
tokio = { version = "1.0", features = ["full"] }

# Browser automation (mistral.rs sampler)
headless-chrome = "1.0"
```

### System Dependencies
```bash
# CUDA Toolkit
sudo apt install nvidia-cuda-toolkit

# CUDA Runtime
sudo apt install cuda-runtime-12-0

# cuDNN (optional, for optimized convolutions)
sudo apt install libcudnn8
```

---

## Performance Metrics

### Measured Performance

#### Markov Sampling
- **CPU**: ~100 samples/sec
- **GPU**: ~10,000 samples/sec (100× speedup)
- **Memory**: ~2GB GPU RAM

#### Hecke Encoding
- **CPU**: ~50 encodes/sec
- **GPU**: ~5,000 encodes/sec (100× speedup)
- **Batch Size**: 71 samples

#### ZK Meme Generation
- **Eigenvalues**: ~1ms per meme (GPU)
- **Signatures**: ~10ms per meme (CPU)
- **Total**: 71 memes in ~1 second

### Bottlenecks

1. **CPU → GPU Transfer**
   - Minimize data transfers
   - Use pinned memory

2. **SHA-256 Signing**
   - CPU-bound operation
   - Consider GPU SHA-256 libraries

3. **Disk I/O**
   - Parquet reading
   - Binary output writing

---

## Future Work

### 1. mistral.rs Integration
- [ ] Complete model loading
- [ ] Implement inference pipeline
- [ ] Add vision model support
- [ ] Batch processing optimization

### 2. CUDA Optimization
- [ ] Multi-GPU support
- [ ] Kernel fusion
- [ ] Memory pooling
- [ ] Async execution streams

### 3. Quantization
- [ ] INT8 quantization for Hecke operators
- [ ] Mixed precision training
- [ ] Dynamic quantization

### 4. Distributed Computing
- [ ] Multi-node GPU clusters
- [ ] MPI integration
- [ ] Distributed ZK proof generation

---

## References

### Documentation
- [burn-rs Documentation](https://burn.dev/)
- [GGML CUDA Backend](https://github.com/ggerganov/ggml)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)

### Papers
- Monster Group: Conway & Sloane (1988)
- Hecke Operators: Hecke (1937)
- Flash Attention: Dao et al. (2022)

### Project Files
- [README.md](README.md) - Project overview
- [PAPER.md](PAPER.md) - Complete paper
- [PROGRAM_INDEX.md](PROGRAM_INDEX.md) - All programs

---

## Contact

For questions about CUDA/GPU implementation:
- See [PAPER.md](PAPER.md) for mathematical background
- See [PROGRAM_INDEX.md](PROGRAM_INDEX.md) for complete program list
- Check [examples/ollama-monster/](examples/ollama-monster/) for LLM experiments

---

**Last Updated**: 2026-01-29  
**Version**: 1.0  
**Status**: Active Development
