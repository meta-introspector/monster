# 📦 Artifact-to-Dataset Mirror

## Overview

Automatically mirrors GitHub Actions build artifacts to HuggingFace datasets.

## Workflow: `artifact-to-dataset.yml`

### Triggers

**Automatic** - After successful completion of:
- Monster Pipeline - Review + ZK-ML + HuggingFace
- Lean Action CI
- Build Monster Walk Paper and WASM
- LMFDB Hecke Analysis Pipeline

**Manual** - Via workflow_dispatch with run ID

### What It Does

```
1. Download artifacts from completed workflow
2. Flatten directory structure
3. Create metadata JSON
4. Upload to HuggingFace (2 repos)
5. Generate summary
```

### Files Mirrored

- `*.parquet` - Review data, performance data
- `*.json` - Metadata, configurations
- `*.md` - Documentation, reviews
- `*.pdf` - Papers, reports
- `*.html` - Literate proofs, visualizations

### Upload Strategy

**Timestamped paths** to avoid overwrites:
```
artifacts/20260129_131500_ci_review.parquet
artifacts/20260129_131500_artifact_metadata.json
```

### Metadata Generated

```json
{
  "workflow_run_id": "21479464693",
  "workflow_name": "Monster Pipeline",
  "commit_sha": "9d6530ff",
  "timestamp": "2026-01-29T13:15:00",
  "files": [
    {"name": "ci_review.parquet", "size": 4812}
  ]
}
```

## HuggingFace Datasets

### 1. introspector/data-moonshine 🌙

**Purpose**: Primary dataset repository

**URL**: https://huggingface.co/datasets/introspector/data-moonshine

### 2. meta-introspector/monster-perf-proofs 👹

**Purpose**: Backup/mirror repository

**URL**: https://huggingface.co/datasets/meta-introspector/monster-perf-proofs

## Usage

### Automatic (Default)

Just push code - artifacts automatically mirror after successful builds!

```bash
git commit -m "Update code"
git push
# → Workflow runs
# → Artifacts generated
# → Mirror workflow triggers
# → Uploads to HuggingFace
```

### Manual Trigger

```bash
# Get run ID
gh run list --limit 5

# Trigger mirror for specific run
gh workflow run "Mirror Artifacts to HuggingFace Datasets" \
  -f run_id=21479464693
```

### Download from HuggingFace

```python
import pandas as pd

# List all files
from huggingface_hub import list_repo_files

files = list_repo_files(
    "introspector/data-moonshine",
    repo_type="dataset"
)
print(files)

# Load specific parquet
df = pd.read_parquet(
    "hf://datasets/introspector/data-moonshine/artifacts/20260129_131500_ci_review.parquet"
)
print(df)
```

## Requirements

**Secret**: `HF_TOKEN`

1. Create token: https://huggingface.co/settings/tokens
2. Add to repo: Settings → Secrets → Actions
3. Name: `HF_TOKEN`
4. Permissions: Write access to datasets

## Workflow Status

```bash
# Check mirror workflow runs
gh run list --workflow="Mirror Artifacts to HuggingFace Datasets"

# View specific run
gh run view <run_id>
```

## Architecture

```
GitHub Actions Workflow
  ↓ (completes successfully)
  ↓ (generates artifacts)
  ↓
Mirror Workflow (triggered)
  ↓ (downloads artifacts)
  ↓ (adds metadata)
  ↓
HuggingFace Datasets
  ├─ introspector/data-moonshine
  └─ meta-introspector/monster-perf-proofs
```

## Benefits

1. **Automatic** - No manual upload needed
2. **Dual backup** - Two repos for redundancy
3. **Timestamped** - No overwrites, full history
4. **Metadata** - Track source workflow and commit
5. **Selective** - Only mirrors successful builds

## Example Flow

```
1. Push code → main branch
2. Monster Pipeline runs (1m33s)
3. Generates ci_review.parquet
4. Uploads as artifact
5. Mirror workflow triggers
6. Downloads artifact
7. Adds metadata
8. Uploads to HuggingFace:
   - artifacts/20260129_131500_ci_review.parquet
   - artifacts/20260129_131500_artifact_metadata.json
9. Available in both datasets!
```

## Summary

✅ **Automatic artifact mirroring**  
✅ **Dual HuggingFace repos**  
✅ **Timestamped uploads**  
✅ **Metadata tracking**  
✅ **Manual trigger option**  

🎯 **Build artifacts automatically become datasets!**
