# 📚 The Stack v2 - Lean4 Code Analysis

**Date**: 2026-01-29  
**Dataset**: bigcode/the-stack-v2 (Lean)  
**Status**: ✅ Analyzed

## The Stack v2 - Lean Dataset

### Overview
- **Total files**: 73,856
- **Total size**: 999 MB (952.9 MB)
- **Avg file size**: 13,529 bytes
- **Median size**: 6,310 bytes
- **Max size**: 6.4 MB

### Star Distribution
- **Avg stars**: 189.3
- **Median stars**: 0
- **Max stars**: 43,494 (sharkdp/bat)

### Top Repositories

| Repository | Stars | Files |
|------------|-------|-------|
| sharkdp/bat | 43,494⭐ | Multiple |
| leanprover/lean4 | 2,827⭐ | Many |

### Sample File Paths
```
/src/algebra/module/basic.lean
/tests/lean/interactive/complete.lean
/src/topology/basic.lean
/library/init/meta/lean/parser.lean
/src/order/filter/basic.lean
```

## Monster Project Comparison

### Monster Lean (32 files)
- **Avg size**: 4,687 bytes
- **Avg lines**: 155
- **Total size**: 150 KB

### The Stack v2 Lean (73,856 files)
- **Avg size**: 13,529 bytes
- **Total size**: 953 MB

### Ratio Analysis

**Monster / Stack v2**: 0.35x

**Interpretation**:
- Monster Lean files are **smaller** than typical Stack v2 files
- Monster: ~4.7 KB avg (focused, specific proofs)
- Stack v2: ~13.5 KB avg (includes large libraries, mathlib)

**Why Monster is smaller**:
1. Focused on Monster Group proofs
2. Specific theorems (8 proven)
3. Not a general library
4. Targeted complexity analysis

**Stack v2 includes**:
1. Full mathlib
2. Large proof libraries
3. General-purpose code
4. Teaching materials

## Key Insights

### 1. Monster is Specialized
- 32 files vs 73,856 in Stack
- Focused on specific domain (Monster Group)
- Smaller, targeted proofs

### 2. Stack v2 is Comprehensive
- Includes leanprover/lean4 official repo
- Full mathlib coverage
- Wide variety of proof styles

### 3. File Size Distribution
```
Monster:  4,687 bytes avg (compact proofs)
Stack v2: 13,529 bytes avg (comprehensive libraries)
```

## Top Monster Lean Files

1. **MusicalPeriodicTable.lean** - 12,276 bytes (421 lines)
2. **ExpressionKernels.lean** - 8,885 bytes (279 lines)
3. **BottPeriodicity.lean** - 8,027 bytes (252 lines)
4. **CrossLanguageComplexity.lean** - 7,948 bytes (248 lines)
5. **MonsterLayers.lean** - 7,690 bytes (257 lines)

## Comparison Summary

| Metric | Monster | Stack v2 | Ratio |
|--------|---------|----------|-------|
| Files | 32 | 73,856 | 0.0004x |
| Avg size | 4,687 | 13,529 | 0.35x |
| Total size | 150 KB | 953 MB | 0.00016x |

## Conclusion

Monster project is a **specialized, focused** Lean4 codebase:
- ✅ Smaller files (easier to understand)
- ✅ Specific domain (Monster Group)
- ✅ 8 proven theorems
- ✅ Cross-language complexity analysis

The Stack v2 provides **comprehensive** Lean ecosystem:
- 📚 73,856 files
- 📚 953 MB of code
- 📚 Includes mathlib, lean4 core
- 📚 Wide variety of proof techniques

**Monster is 0.35x the size of typical Stack v2 Lean files** - indicating focused, efficient proofs rather than general libraries.

## Files Generated

- `stack_v2_lean_sample.parquet` - 1,000 sample files
- Analysis includes: repo_name, length_bytes, star_events_count, path

## Next Steps

1. **Compare proof techniques** - Monster vs mathlib
2. **Analyze imports** - What libraries do we use?
3. **Benchmark complexity** - Monster theorems vs Stack theorems
4. **Upload analysis** to HuggingFace datasets
