# ✅ GitHub Actions - COMPLETE

## Workflow Created

**File**: `.github/workflows/monster-pipeline.yml`

## Pipeline Steps

```yaml
1. Checkout code
2. Setup Python 3.10
3. Install: pandas, pyarrow, huggingface_hub
4. Setup Lean4
5. Build Lean4 proofs
6. Review HTML proof (if exists)
7. Generate ZK-ML proof (if script exists)
8. Generate review parquet (9 personas)
9. Upload to HuggingFace (2 repos)
10. Upload GitHub artifacts
11. Generate summary
```

## Triggers

- ✅ Push to `main` or `master`
- ✅ Pull requests
- ✅ Manual workflow dispatch

## Required Secret

**Name**: `HF_TOKEN`  
**Get from**: https://huggingface.co/settings/tokens  
**Add at**: Repository Settings → Secrets → Actions

## Uploads To

1. **introspector/data-moonshine** 🌙
2. **meta-introspector/monster-perf-proofs** 👹

## Files Uploaded

```
precommit_review.parquet (9 rows, 9 personas)
zkml_witness.parquet (if generated)
commit_*.parquet (if generated)
```

## Artifacts Saved

```
*.parquet
*.json
*_REVIEW.md
ZKML_*.md
```

**Retention**: 30 days

## Next Steps

### 1. Add Secret

```bash
# Go to repo settings
https://github.com/YOUR_USERNAME/monster-lean/settings/secrets/actions

# Add new secret
Name: HF_TOKEN
Value: hf_...
```

### 2. Test Workflow

```bash
# Commit and push
git add .github/workflows/monster-pipeline.yml
git commit -m "Add Monster pipeline workflow"
git push

# Watch at
https://github.com/YOUR_USERNAME/monster-lean/actions
```

### 3. Verify Upload

```python
import pandas as pd

# Check uploaded data
df = pd.read_parquet(
    "hf://datasets/introspector/data-moonshine/precommit_review.parquet"
)
print(df)
```

## Summary

✅ **Workflow file created**  
✅ **9 pipeline steps**  
✅ **2 HuggingFace repos**  
✅ **Automatic on push/PR**  
✅ **Artifacts saved 30 days**  

🔧 **GitHub Actions ready to go!**
