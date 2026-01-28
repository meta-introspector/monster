# Vision Model Verification Pipeline

## Objective
Close the loop: Generate documents about Monster group → Use vision model to read them → Trace registers → Verify Monster primes appear in both content and computation.

## Pipeline

```
1. Generate Documents
   ├── PDF: Monster Walk results
   ├── PNG: Prime emoji periodic table
   └── PNG: Error correction code mappings

2. Vision Model Reads Documents
   ├── Extract text from PDF (OCR)
   ├── Detect patterns in images
   └── Identify Monster primes mentioned

3. Trace Vision Model Inference
   ├── Feed document to vision model
   ├── Capture CPU registers with perf
   └── Measure prime divisibility

4. Verify Correspondence
   ├── Primes in document content
   ├── Primes in register values
   └── Prove: Model internalizes what it reads
```

## Implementation Plan

### Phase 1: Document Generation
```bash
# Generate PDF from our results
cd examples/ollama-monster
pandoc RESULTS.md -o RESULTS.pdf

# Generate PNG visualizations
cargo run --bin visualize-primes
```

### Phase 2: Vision Model Setup
```bash
# Use LLaVA or similar local vision model
# Options:
# - llama.cpp with LLaVA
# - Ollama with llava:7b
# - Qwen-VL
```

### Phase 3: Vision Tracing
```bash
# Trace vision model reading our documents
./trace_vision.sh RESULTS.pdf
./trace_vision.sh prime_table.png
```

### Phase 4: Analysis
```bash
# Compare document content vs register patterns
cargo run --bin verify-correspondence
```

## Expected Results

### Document Content
- RESULTS.pdf mentions: primes 2, 3, 5, 7, 11
- prime_table.png shows: all 15 Monster primes
- CODE_MONSTER_MAP.json contains: 982 codes with prime signatures

### Register Patterns (when vision model reads documents)
- Reading RESULTS.pdf → 80% prime 2, 49% prime 3
- Reading prime_table.png → Higher primes activated (13, 17, 47)
- Reading CODE_MONSTER_MAP.json → All 5 top primes present

### Verification
**Hypothesis**: When vision model reads document about prime P, register values show increased divisibility by P.

**Test**: 
1. Create document emphasizing prime 47
2. Vision model reads it
3. Measure: Does prime 47 resonance increase?

## Tools Needed

### Vision Models (Local)
- [ ] LLaVA 7B (via llama.cpp)
- [ ] Qwen-VL (via Ollama)
- [ ] OpenCV for image preprocessing

### Document Processing
- [ ] PyMuPDF for PDF text extraction
- [ ] Tesseract OCR for image text
- [ ] PIL/Pillow for image manipulation

### Integration
- [ ] `trace_vision.sh` - Trace vision model with perf
- [ ] `extract_content.py` - Extract primes from documents
- [ ] `verify_correspondence.rs` - Compare content vs registers

## Current Status

✅ **Completed**:
- Text model tracing (qwen2.5:3b)
- Register analysis tools
- Document generation (markdown)

🔄 **In Progress**:
- PDF/PNG generation
- Vision model setup

❌ **TODO**:
- Vision model tracing
- Content extraction
- Correspondence verification

## Next Steps

1. Generate visual documents
2. Set up local vision model
3. Implement trace_vision.sh
4. Run verification experiments
5. Document results

This closes the loop: **Documents → Vision → Registers → Verification**
