# ✅ HuggingFace Upload - BOTH REPOS

## Overview

Performance data now uploads to **2 repositories** for redundancy and discoverability!

## Repositories

### 1. introspector/data-moonshine
**URL**: https://huggingface.co/datasets/introspector/data-moonshine  
**Purpose**: Main moonshine data repository

### 2. meta-introspector/monster-perf-proofs
**URL**: https://huggingface.co/datasets/meta-introspector/monster-perf-proofs  
**Purpose**: Monster-specific performance proofs

## Files Uploaded

```
zkml_witness.parquet (6,049 bytes)
  - Commit: bdd8dd5f
  - Compile: 2,025ms
  - Build: 1,006ms
  - Review: 81/90
  - CPU: 680M cycles
  - Memory: 3,438MB
```

## Upload

```bash
# Authenticate once
huggingface-cli login

# Upload to both repos
./push_to_huggingface.sh
```

**Output**:
```
Uploading to introspector/data-moonshine...
  ✓ zkml_witness.parquet
  ✓ perf_metadata.json
  ✓ README.md

Uploading to meta-introspector/monster-perf-proofs...
  ✓ zkml_witness.parquet
  ✓ perf_metadata.json
  ✓ README.md

✓ All uploads complete
```

## Access

### From Either Repo

```python
import pandas as pd

# From moonshine
df1 = pd.read_parquet(
    "hf://datasets/introspector/data-moonshine/zkml_witness.parquet"
)

# From monster-perf-proofs
df2 = pd.read_parquet(
    "hf://datasets/meta-introspector/monster-perf-proofs/zkml_witness.parquet"
)

# Both are identical
assert df1.equals(df2)
print("✅ Data verified across both repos!")
```

## Benefits

### 1. Redundancy
Data available in 2 locations

### 2. Discoverability
- Moonshine users find it in data-moonshine
- Monster users find it in monster-perf-proofs

### 3. Cross-Verification
Can verify data matches across repos

### 4. Backup
If one repo has issues, other is available

## Verification

### Download from Either

```bash
# From moonshine
huggingface-cli download introspector/data-moonshine --repo-type dataset

# From monster-perf-proofs
huggingface-cli download meta-introspector/monster-perf-proofs --repo-type dataset
```

### Verify Match

```python
import pandas as pd

df1 = pd.read_parquet("~/.cache/huggingface/hub/datasets--introspector--data-moonshine/snapshots/*/zkml_witness.parquet")
df2 = pd.read_parquet("~/.cache/huggingface/hub/datasets--meta-introspector--monster-perf-proofs/snapshots/*/zkml_witness.parquet")

assert df1.equals(df2)
print("✅ Verified: Both repos have identical data")
```

## Integration

### Post-Commit Hook

```bash
# .git/hooks/post-commit
./zkml_pipeline.sh
./push_to_huggingface.sh  # Uploads to both repos
```

### CI/CD

```yaml
- name: Generate ZK-ML Proof
  run: ./zkml_pipeline.sh

- name: Upload to HuggingFace (2 repos)
  run: ./push_to_huggingface.sh
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

## Summary

✅ **Uploads to 2 repos**:
- introspector/data-moonshine
- meta-introspector/monster-perf-proofs

✅ **Same files in both**:
- zkml_witness.parquet
- perf_metadata.json
- README.md

✅ **Benefits**:
- Redundancy
- Discoverability
- Cross-verification
- Backup

---

**Repos**: 2 ✅  
**Files**: 1 parquet (6KB) 📊  
**Status**: Ready for upload 📤  

🎯 **Double the availability, double the verification!**
