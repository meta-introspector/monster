# 📊 Lean4 Metadata Analysis & Expansion Recommendations

**Date**: 2026-01-29  
**Analysis**: The Stack v2 Lean dataset  
**Goal**: Expand Monster project with group theory, topology, and math

## Current State

### On Disk
- **Parquet file**: 13.0 MB (compressed)
- **Uncompressed**: 952.9 MB
- **Compression ratio**: 73.4x 🎉
- **Total files**: 73,856

### Monster Project (Current)
- **Files**: 32 Lean files
- **Size**: 0.15 MB
- **Focus**: Monster Group proofs

## Topic Analysis

### Group Theory
- **Files**: 11,282 (15.3% of dataset)
- **Size**: 176.2 MB
- **Key repos**: mathlib, formalising-mathematics

### Topology
- **Files**: 4,916 (6.7% of dataset)
- **Size**: 102.6 MB
- **Key repos**: mathlib, lean4

### Math (General)
- **Files**: 13,496 (18.3% of dataset)
- **Size**: 188.9 MB
- **Key repos**: mathlib, lean4

## 🏆 Top 3 Repos for Expansion

### 1. leanprover-community/mathlib ⭐⭐⭐

**Why**: The definitive Lean math library

**Stats**:
- Stars: 1,595⭐
- Files: 3,462
- Size: 44.1 MB
- Avg file: 13,363 bytes

**Relevant Content**:
- Group theory files: 870
- Topology files: 313
- Algebra, ring theory, category theory

**Sample Paths**:
```
/src/algebra/char_p/quotient.lean
/src/ring_theory/ring_hom_properties.lean
/src/group_theory/sylow.lean
/src/topology/basic.lean
```

**Gain**: 44.1 MB (294x current size)

**Recommendation**: ⭐⭐⭐ ESSENTIAL
- Most comprehensive group theory
- Includes sporadic groups
- Topology foundations
- Well-documented

### 2. leanprover/lean4 ⭐⭐

**Why**: Official Lean4 core + examples

**Stats**:
- Stars: 2,827⭐ (highest!)
- Files: 2,653
- Size: 8.6 MB
- Avg file: 3,409 bytes

**Relevant Content**:
- Group theory files: 0 (core only)
- Topology files: 1
- Core tactics, metaprogramming

**Sample Paths**:
```
/src/Lean/HeadIndex.lean
/tests/lean/run/reduce2.lean
/src/Lean/Meta/Tactic/Simp.lean
```

**Gain**: 8.6 MB (57x current size)

**Recommendation**: ⭐⭐ USEFUL
- Learn advanced tactics
- Metaprogramming techniques
- Official examples
- Not focused on math

### 3. ImperialCollegeLondon/formalising-mathematics ⭐⭐⭐

**Why**: Teaching-focused, accessible proofs

**Stats**:
- Stars: 284⭐
- Files: 45
- Size: 0.4 MB
- Avg file: 10,300 bytes

**Relevant Content**:
- Group theory files: 4
- Topology files: 4
- Teaching materials

**Sample Paths**:
```
/src/week_7/Part_C_back_to_Z.lean
/src/week_8/ideas/Part_D_boundary_map.lean
/src/group_theory/group_basics.lean
```

**Gain**: 0.4 MB (2.7x current size)

**Recommendation**: ⭐⭐⭐ EXCELLENT FOR LEARNING
- Clear, pedagogical proofs
- Group theory basics
- Topology introduction
- Small, digestible

## Potential Gain Summary

| Repo | Files | Size | Gain Ratio |
|------|-------|------|------------|
| mathlib | 3,462 | 44.1 MB | 294x |
| lean4 | 2,653 | 8.6 MB | 57x |
| formalising-mathematics | 45 | 0.4 MB | 2.7x |
| **TOTAL** | **6,160** | **53.2 MB** | **355x** |

**Current Monster**: 32 files, 0.15 MB  
**After expansion**: 6,192 files, 53.35 MB  
**Growth**: 355x size, 193x files

## Compression Efficiency

**Amazing compression**: 73.4x ratio!

- Download: 13 MB parquet
- Get: 953 MB of Lean code
- Storage efficient: Keep compressed

## Specific Recommendations

### For Monster Group Theory

**From mathlib**:
```
/src/group_theory/sylow.lean
/src/group_theory/perm/sign.lean
/src/group_theory/specific_groups/cyclic.lean
/src/algebra/group/basic.lean
```

**Why**: Sylow theorems, permutation groups, specific groups

### For Topology

**From mathlib**:
```
/src/topology/basic.lean
/src/topology/algebra/group.lean
/src/topology/instances/real.lean
```

**Why**: Topological groups, foundations

### For Learning

**From formalising-mathematics**:
```
/src/week_7/Part_C_back_to_Z.lean
/src/group_theory/group_basics.lean
```

**Why**: Clear teaching examples

## Implementation Strategy

### Phase 1: Core (mathlib group theory)
- Download: 870 files, ~10.6 MB
- Focus: Group theory essentials
- Time: 1 week

### Phase 2: Topology (mathlib topology)
- Download: 313 files, ~5.5 MB
- Focus: Topological foundations
- Time: 1 week

### Phase 3: Learning (formalising-mathematics)
- Download: 45 files, 0.4 MB
- Focus: Pedagogical examples
- Time: 3 days

### Phase 4: Advanced (lean4 core)
- Download: 2,653 files, 8.6 MB
- Focus: Advanced tactics
- Time: 2 weeks

## Storage Requirements

**Current**: 0.15 MB  
**After Phase 1**: 10.75 MB (72x)  
**After Phase 2**: 16.25 MB (108x)  
**After Phase 3**: 16.65 MB (111x)  
**After Phase 4**: 25.25 MB (168x)  
**Full expansion**: 53.35 MB (355x)

**Disk space needed**: ~60 MB (with overhead)

## Files Generated

- `lean_metadata_analysis.json` - Full analysis
- `lean_repo_recommendations.parquet` - Top 3 repos data

## Next Steps

1. **Clone mathlib** group theory subset
2. **Study formalising-mathematics** for patterns
3. **Extract relevant proofs** for Monster
4. **Integrate topology** for spectral analysis
5. **Document learnings** in Monster proofs

## Conclusion

**Best ROI**: leanprover-community/mathlib
- 870 group theory files
- 313 topology files
- 44.1 MB (294x gain)
- Most relevant to Monster project

**Best for learning**: ImperialCollegeLondon/formalising-mathematics
- 45 files (manageable)
- Teaching-focused
- Clear examples

**Best for tactics**: leanprover/lean4
- 2,653 files
- Official examples
- Advanced techniques

**Recommended order**: mathlib → formalising-mathematics → lean4
