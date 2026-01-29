# 🚀 Commit + Push Script

## Overview

**One script** to commit to git AND push to both HuggingFace repos!

## Usage

### Basic

```bash
./commit_and_push.sh
```

Uses default message: `Update: 2026-01-29_08:04:46`

### With Custom Message

```bash
./commit_and_push.sh "Add new feature"
```

### What It Does

```
📝 [1/3] Git commit
  - Stage all changes (if nothing staged)
  - Commit with message
  - Get commit hash

💾 [2/3] Generate parquet
  - Copy precommit_review.parquet → commit_HASH.parquet
  - Prepare zkml_witness.parquet

📤 [3/3] Push to HuggingFace
  - Upload to introspector/data-moonshine
  - Upload to meta-introspector/monster-perf-proofs
  - All parquet files
```

## Example Run

```bash
$ ./commit_and_push.sh "Prove Coq ≃ Lean4 ≃ Rust"

🚀 COMMIT + PUSH TO BOTH REPOS
============================================================

📝 [1/3] Git commit...
✓ Staged files: 3
✓ Committed: abc123de

💾 [2/3] Generating parquet from commit...
✓ commit_abc123de.parquet
✓ zkml_witness.parquet ready

📤 [3/3] Pushing to HuggingFace (2 repos)...
✓ Authenticated

Uploading to introspector/data-moonshine...
  → zkml_witness.parquet
  → precommit_review.parquet
  → commit_abc123de.parquet
✓ introspector/data-moonshine complete

Uploading to meta-introspector/monster-perf-proofs...
  → zkml_witness.parquet
  → precommit_review.parquet
  → commit_abc123de.parquet
✓ meta-introspector/monster-perf-proofs complete

✅ All uploads complete!

✅ COMMIT + PUSH COMPLETE
============================================================

Git:
  Commit: abc123de
  Message: Prove Coq ≃ Lean4 ≃ Rust

HuggingFace:
  ✓ introspector/data-moonshine
  ✓ meta-introspector/monster-perf-proofs

Files:
  ✓ zkml_witness.parquet
  ✓ precommit_review.parquet
  ✓ commit_abc123de.parquet

🔗 View at:
  https://huggingface.co/datasets/introspector/data-moonshine
  https://huggingface.co/datasets/meta-introspector/monster-perf-proofs
```

## Files Uploaded

### 1. zkml_witness.parquet
- ZK-ML proof witness
- Performance metrics
- Constraint verification

### 2. precommit_review.parquet
- 9-persona review scores
- HTML/docs/code metrics
- Per-commit review data

### 3. commit_HASH.parquet
- Copy of precommit review
- Named by commit hash
- Historical record

## Requirements

### Authentication

```bash
# One-time setup
huggingface-cli login
# Enter your token
```

### Installation

```bash
pip install huggingface_hub
```

## Integration

### Manual Workflow

```bash
# Make changes
vim file.rs

# Commit + push to both repos
./commit_and_push.sh "Add feature"
```

### Automated Workflow

```bash
# In CI/CD
- name: Commit and Push
  run: ./commit_and_push.sh "CI: ${{ github.sha }}"
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

### Post-Commit Hook

```bash
# .git/hooks/post-commit
#!/bin/bash
./commit_and_push.sh "Auto-push: $(git log -1 --pretty=%B)"
```

## Error Handling

### Not Authenticated

```
❌ Not authenticated
   Run: huggingface-cli login
```

**Fix**: Run `huggingface-cli login`

### No Files to Upload

```
⚠️  No parquet files to upload
```

**Fix**: Run `./zkml_pipeline.sh` first

### Upload Failed

```
❌ Upload failed
```

**Fix**: Check network, check repo permissions

## Benefits

### 1. Single Command
One script does everything

### 2. Atomic Operation
Commit + push together

### 3. Dual Upload
Both repos updated simultaneously

### 4. Historical Record
Each commit gets its own parquet file

### 5. Automatic
No manual steps needed

## Verification

### Check Git

```bash
git log -1
```

### Check HuggingFace

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

# Verify match
assert df1.equals(df2)
print("✅ Both repos have identical data!")
```

## Summary

✅ **One script does it all**:
- Git commit
- Generate parquet
- Push to 2 HuggingFace repos

✅ **Simple usage**:
```bash
./commit_and_push.sh "Your message"
```

✅ **Complete automation**:
- No manual steps
- Atomic operation
- Dual redundancy

---

**Script**: commit_and_push.sh ✅  
**Repos**: 2 (moonshine + monster-perf-proofs) 📤  
**Files**: 3 parquet files per commit 📊  

🚀 **One command to rule them all!**
