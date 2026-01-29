# 🎵 Harmonic Analysis Nix Flakes

**Date**: 2026-01-29  
**Location**: `harmonics_repos/flake.nix`  
**Status**: ✅ Complete

## Flake Structure

### Three Development Shells

#### 1. `dftk` - DFTK.jl Environment ⭐⭐⭐

**Focus**: Spherical harmonics, FFT, symmetry

**Includes**:
- Julia with FFTW, AbstractFFTs, LinearAlgebra, StaticArrays
- FFTW libraries (float and double)
- pkg-config

**Key files**:
- `src/common/spherical_harmonics.jl` ⭐⭐⭐⭐⭐
- `src/fft.jl` ⭐⭐⭐
- `src/symmetry.jl` ⭐⭐⭐

**Usage**:
```bash
cd harmonics_repos
nix develop .#dftk
cd DFTK.jl
julia
> include("src/common/spherical_harmonics.jl")
> ylm_real(2, 0, [0.0, 0.0, 1.0])  # Test d_z² orbital
```

#### 2. `approxfun` - ApproxFun.jl Environment ⭐⭐

**Focus**: Spectral methods, generic FFT

**Includes**:
- Julia with packages
- FFTW libraries

**Key files**:
- `src/Extras/fftGeneric.jl` ⭐⭐⭐
- `examples/Eigenvalue_anharmonic.jl` ⭐⭐⭐

**Usage**:
```bash
nix develop .#approxfun
cd ApproxFun.jl
julia
```

#### 3. `default` - Combined Environment ⭐⭐⭐

**Focus**: Both repos + analysis tools

**Includes**:
- Everything from dftk and approxfun
- Python3 with pandas, pyarrow, numpy
- Analysis tools

**Usage**:
```bash
nix develop
# Access both repos + Python analysis
```

## Apps

### 1. `test-spherical` - Test Spherical Harmonics

**What it does**:
- Tests Y_l^m for l=0,1,2 (s, p, d orbitals)
- Evaluates at [0,0,1] (z-axis)
- Verifies implementation

**Usage**:
```bash
cd harmonics_repos
nix run .#test-spherical
```

**Expected output**:
```
Testing Spherical Harmonics Y_l^m
==================================

s orbital (l=0, m=0):
  Y_0^0([0,0,1]) = 0.2820947917738781

p orbitals (l=1):
  Y_1^-1([0,0,1]) = 0.0
  Y_1^0([0,0,1]) = 0.4886025119029199
  Y_1^1([0,0,1]) = 0.0

d orbitals (l=2):
  Y_2^-2([0,0,1]) = 0.0
  Y_2^-1([0,0,1]) = 0.0
  Y_2^0([0,0,1]) = 0.6307831305050401
  Y_2^1([0,0,1]) = 0.0
  Y_2^2([0,0,1]) = 0.0

✅ Spherical harmonics working!
```

### 2. `test-fft` - Test FFT

**What it does**:
- Tests basic FFT/IFFT
- Verifies FFTW installation

**Usage**:
```bash
nix run .#test-fft
```

**Expected output**:
```
Testing FFT
===========

Input: [1.0, 2.0, 3.0, 4.0]
FFT: [10.0+0.0im, -2.0+2.0im, -2.0+0.0im, -2.0-2.0im]
IFFT: [1.0+0.0im, 2.0+0.0im, 3.0+0.0im, 4.0+0.0im]

✅ FFT working!
```

## Quick Start

### Enter DFTK Environment

```bash
cd harmonics_repos
nix develop .#dftk
```

### Test Spherical Harmonics

```bash
nix run .#test-spherical
```

### Test FFT

```bash
nix run .#test-fft
```

### Enter Combined Environment

```bash
nix develop
```

### Analyze Rankings

```bash
nix develop
python3 -c 'import pandas as pd; print(pd.read_parquet("../harmonics_ranked.parquet").head())'
```

## Connection to Monster Group

### Spherical Harmonics = Group Representations

**SO(3) (continuous group)**:
```julia
# Y_l^m: Spherical harmonics
# l = 0: 1 function (s orbital)
# l = 1: 3 functions (p orbitals)
# l = 2: 5 functions (d orbitals)
# Dimension: 2l+1
```

**Monster (finite group)**:
```lean
-- χ_i: Characters
-- 194 irreducible representations
-- Various dimensions
```

### FFT = Fourier Transform on Groups

**Continuous (SO(3))**:
```julia
# FFT on sphere
f̂(l,m) = ∫ f(θ,φ) Y_l^m(θ,φ) sin(θ) dθ dφ
```

**Finite (Monster)**:
```lean
-- Fourier transform on group
f̂(χ) = (1/|G|) Σ_{g∈G} f(g) χ(g)
```

### Orthogonality Relations

**SO(3)**:
```julia
∫ Y_l^m Y_l'^m' = δ_ll' δ_mm'
```

**Monster**:
```lean
Σ_g χ_i(g) χ_j(g) = |G| δ_ij
```

## Files in Flake

```
harmonics_repos/
├── flake.nix ⭐⭐⭐
├── DFTK.jl/
│   ├── src/
│   │   ├── common/
│   │   │   └── spherical_harmonics.jl ⭐⭐⭐⭐⭐
│   │   ├── fft.jl ⭐⭐⭐
│   │   └── symmetry.jl ⭐⭐⭐
│   └── ...
└── ApproxFun.jl/
    ├── src/
    │   └── Extras/
    │       └── fftGeneric.jl ⭐⭐⭐
    └── examples/
        └── Eigenvalue_anharmonic.jl ⭐⭐⭐
```

## Next Steps

### 1. Test Spherical Harmonics ⭐⭐⭐

```bash
nix run .#test-spherical
```

**Goal**: Verify Y_l^m implementation works

### 2. Study Implementation ⭐⭐⭐

```bash
nix develop .#dftk
cd DFTK.jl
julia
> include("src/common/spherical_harmonics.jl")
# Study the code
```

**Goal**: Understand how Y_l^m is computed

### 3. Test with Small Groups ⭐⭐

```julia
# Create test for S5 (symmetric group)
function s5_character_transform(f, χ)
    # Implement for S5 (120 elements)
end
```

**Goal**: Adapt for finite groups

### 4. Implement for Monster ⭐⭐⭐

```lean
-- MonsterLean/SphericalHarmonics.lean
-- Port Julia code to Lean4
def monsterCharacterTransform (f : Monster → ℂ) (χ : Character Monster) : ℂ
```

**Goal**: Apply to Monster (194 characters)

## Summary

✅ **3 development shells** (dftk, approxfun, default)  
✅ **2 test apps** (test-spherical, test-fft)  
✅ **Julia + FFTW** fully configured  
✅ **Python analysis tools** included  
✅ **Ready to test** spherical harmonics  
✅ **Ready to adapt** for Monster group

**Nix flakes created for harmonic analysis!** 🎵✅
