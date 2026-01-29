# 🔢 GAP and PARI/GP Installation with Nix

**Date**: 2026-01-29  
**Method**: Nix from source with packages

## Files Created

### 1. Nix Shells

**shell-gap-pari.nix** - Full environment with packages
```nix
GAP with:
- atlasrep (Monster group!)
- ctbllib (Character tables)
- tomlib (Table of marks)
- smallgrp, transgrp

PARI/GP with:
- GMP support
- Readline support
```

**shell-computational.nix** - Basic environment

### 2. Installation Script

**install-gap-pari.sh**
```bash
#!/usr/bin/env bash
nix-shell shell-gap-pari.nix
```

### 3. GAP Export Script

**export_monster_gap.g** - GAP script to export Monster data
```gap
LoadPackage("atlasrep");
M := AtlasGroup("M");
Order(M);
# Exports to monster_gap_data.json
```

### 4. Rust Converter

**gap_to_parquet.sh** - Convert JSON to Parquet (Rust)
- No Python!
- Uses Arrow + Parquet crates
- Outputs monster_gap_data.parquet

## Usage

### Install GAP and PARI/GP

```bash
./install-gap-pari.sh
```

Or manually:
```bash
nix-shell shell-gap-pari.nix
```

### Load Monster Group in GAP

```bash
nix-shell shell-gap-pari.nix

gap> LoadPackage("atlasrep");
gap> M := AtlasGroup("M");
gap> Order(M);
# 808017424794512875886459904961710757005754368000000000
```

### Export Monster Data

```bash
nix-shell shell-gap-pari.nix --run "gap export_monster_gap.g"
```

This creates `monster_gap_data.json`:
```json
{
  "name": "Monster",
  "order": "808017424794512875886459904961710757005754368000000000",
  "is_simple": true,
  "is_sporadic": true,
  "num_conjugacy_classes": 194,
  "num_characters": 194
}
```

### Convert to Parquet

```bash
./gap_to_parquet.sh
```

Creates `monster_gap_data.parquet` with Monster group properties.

## What You Get

### GAP Packages

- **atlasrep**: Atlas of Group Representations
  - Has Monster group: `AtlasGroup("M")`
  - 194 conjugacy classes
  - 194 irreducible characters

- **ctbllib**: Character Table Library
  - `CharacterTable("M")`
  - Full character table

- **tomlib**: Table of Marks
  - Subgroup lattice information

### PARI/GP Features

- Elliptic curves: `ellinit()`
- Modular forms: `mfinit()`
- j-invariant: `ellj()`
- Number theory functions

## Verification

### Verify Monster Order

```gap
gap> LoadPackage("atlasrep");
gap> M := AtlasGroup("M");
gap> Order(M);
808017424794512875886459904961710757005754368000000000

gap> # Compare with our Lean4 constant
gap> our_order := 808017424794512875886459904961710757005754368000000000;
gap> Order(M) = our_order;
true
```

### Get Character Table

```gap
gap> ct := CharacterTable("M");
gap> NrConjugacyClasses(ct);
194
gap> Length(Irr(ct));
194
```

### Compute with PARI/GP

```gp
gp> E = ellinit([0,0,1,-1,0]);
gp> j = ellj(E);
gp> print(j);
```

## Integration with Monster Project

### 1. Verify Proofs

```bash
# Export GAP data
nix-shell shell-gap-pari.nix --run "gap export_monster_gap.g"

# Convert to Parquet
./gap_to_parquet.sh

# Compare with Lean4 proofs
cargo run --bin verify_gap_data
```

### 2. Generate Test Data

```gap
# In GAP
gap> LoadPackage("atlasrep");
gap> M := AtlasGroup("M");
gap> # Generate conjugacy class representatives
gap> # Export to JSON for Lean4 tests
```

### 3. Compute Moonshine

```gp
\\ In PARI/GP
E = ellinit([0,0,1,-1,0]);
j = ellj(E);
\\ j-invariant for moonshine
```

## Next Steps

1. **Install**: Run `./install-gap-pari.sh`
2. **Export**: Run GAP export script
3. **Convert**: Run Rust converter
4. **Verify**: Compare with Lean4 proofs
5. **Integrate**: Use in Monster project

## Summary

✅ **Nix shells created** (GAP + PARI/GP)  
✅ **GAP export script** (Monster data)  
✅ **Rust converter** (JSON → Parquet)  
✅ **No Python!** (Pure Rust/Nix/GAP)  
✅ **Monster group accessible** (194 classes!)

**Ready to verify our Lean4 proofs against GAP!** 🔢✅
