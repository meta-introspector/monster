# 📊 Monster Project Code Benchmark

**Date**: 2026-01-29  
**Status**: ✅ Benchmarked

## Summary

- **Total Files**: 50
- **Total Size**: 255,083 bytes (249.1 KB)
- **Total Lines**: 7,897
- **Languages**: Rust (18), Lean (32)

## Metrics

### Overall
| Metric | Value |
|--------|-------|
| Avg file size | 5,101 bytes |
| Avg lines/file | 158 |
| Median file size | 5,094 bytes |
| Median lines | 160 |

### By Language

**Rust** (18 files)
- Avg size: 5,838 bytes
- Avg lines: 162

**Lean** (32 files)
- Avg size: 4,687 bytes
- Avg lines: 155

## Top 5 Largest Files

1. **MusicalPeriodicTable.lean** - 12,276 bytes, 421 lines
2. **musical_periodic_table.rs** - 11,652 bytes, 218 lines
3. **group_harmonics.rs** - 10,661 bytes, 246 lines
4. **review_paper.rs** - 9,633 bytes, 287 lines
5. **ExpressionKernels.lean** - 8,885 bytes, 279 lines

## Comparison Notes

### The Stack v2 Access

The Stack v2 dataset is **gated** and requires authentication:
- Dataset: `bigcode/the-stack-v2`
- Size: 3+ trillion tokens
- Status: Requires HuggingFace access request

To access:
1. Visit https://huggingface.co/datasets/bigcode/the-stack-v2
2. Request access
3. Login: `huggingface-cli login`
4. Re-run analysis

### Industry Benchmarks (Estimated)

Based on typical open-source projects:

| Metric | Monster | Typical OSS | Ratio |
|--------|---------|-------------|-------|
| Avg file size | 5,101 bytes | ~3,000 bytes | 1.7x |
| Avg lines | 158 | ~100 | 1.6x |

**Interpretation**: Monster files are ~1.6-1.7x larger than typical, indicating:
- More comprehensive implementations
- Detailed documentation
- Complex formal proofs

## Files Generated

- `monster_code_analysis.parquet` - Full analysis (50 rows)
- `code_benchmark.json` - Summary metrics

## Usage

### View Analysis

```python
import pandas as pd

df = pd.read_parquet('monster_code_analysis.parquet')
print(df[['file', 'language', 'size', 'lines']])
```

### Query Specific Language

```python
rust_files = df[df['language'] == 'Rust']
print(f"Rust avg: {rust_files['size'].mean():.0f} bytes")

lean_files = df[df['language'] == 'Lean']
print(f"Lean avg: {lean_files['size'].mean():.0f} bytes")
```

### Find Large Files

```python
large = df[df['size'] > 8000]
print(large[['file', 'size', 'lines']])
```

## Next Steps

1. **Request Stack v2 access** for full comparison
2. **Upload benchmark** to HuggingFace datasets
3. **Track over time** to monitor code growth
4. **Compare with similar projects** (formal verification, group theory)

## Uploaded To

- `introspector/data-moonshine`
- `meta-introspector/monster-perf-proofs`

Paths:
- `benchmarks/monster_code_analysis.parquet`
- `benchmarks/code_benchmark.json`
