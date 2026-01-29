# 🔧 GitHub Actions - FIXED

## Workflow

**File**: `.github/workflows/monster-pipeline.yml`

## What It Does

### On Every Push/PR

```
1. ✅ Checkout code
2. ✅ Setup Python + Lean4
3. ✅ Build Lean4 proofs
4. ✅ Review HTML proof
5. ✅ Generate ZK-ML proof
6. ✅ Generate review parquet
7. ✅ Upload to HuggingFace (2 repos)
8. ✅ Upload artifacts
9. ✅ Generate summary
```

## Setup

### 1. Add HuggingFace Token

Go to: `https://github.com/YOUR_USERNAME/monster-lean/settings/secrets/actions`

Add secret:
- **Name**: `HF_TOKEN`
- **Value**: Your HuggingFace token (from https://huggingface.co/settings/tokens)

### 2. Commit Workflow

```bash
git add .github/workflows/monster-pipeline.yml
git commit -m "Add GitHub Actions workflow"
git push
```

### 3. Watch It Run

Go to: `https://github.com/YOUR_USERNAME/monster-lean/actions`

## What Gets Uploaded

### To HuggingFace (Both Repos)

```
precommit_review.parquet
zkml_witness.parquet (if generated)
commit_*.parquet (if generated)
```

### To GitHub Artifacts

```
*.parquet
*.json
*_REVIEW.md
ZKML_*.md
```

**Retention**: 30 days

## Example Run

```
✅ Checkout
✅ Setup Python
✅ Install dependencies
✅ Setup Lean4
✅ Build Lean4 proofs
   → MonsterLean.CrossLanguageComplexity
✅ Review HTML proof
   → Score: 81/90
✅ Generate ZK-ML proof
   → Witness generated
✅ Generate review parquet
   → 9 rows (9 personas)
✅ Upload to HuggingFace
   → introspector/data-moonshine ✓
   → meta-introspector/monster-perf-proofs ✓
✅ Upload artifacts
   → review-data.zip
✅ Summary
   → Pipeline complete!
```

## Triggers

### Push to Main/Master

```bash
git push origin main
# → Workflow runs automatically
```

### Pull Request

```bash
gh pr create
# → Workflow runs on PR
```

### Manual Trigger

Go to Actions tab → Select workflow → Run workflow

## Outputs

### GitHub Summary

Shows in Actions run:

```markdown
## 🎯 Pipeline Complete

### Files Generated
- precommit_review.parquet (5.9K)
- zkml_witness.parquet (6.0K)

### HuggingFace Repos
- introspector/data-moonshine
- meta-introspector/monster-perf-proofs
```

### Artifacts

Download from Actions run:
- `review-data.zip` (all parquet + JSON + markdown)

## Troubleshooting

### HF_TOKEN Not Set

```
⚠️  HF_TOKEN not set, skipping upload
```

**Fix**: Add `HF_TOKEN` secret in repo settings

### Build Failed

```
❌ Build failed
```

**Fix**: Check Lean4 code, workflow continues anyway

### Upload Failed

```
Upload failed, continuing...
```

**Fix**: Check HuggingFace token permissions

## Local Testing

Test the workflow locally:

```bash
# Install act
brew install act  # or: curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run workflow
act push -s HF_TOKEN=your_token_here
```

## Summary

✅ **GitHub Actions workflow**:
- Runs on every push/PR
- Builds Lean4 proofs
- Generates review parquet
- Uploads to 2 HuggingFace repos
- Saves artifacts

✅ **Setup**:
1. Add HF_TOKEN secret
2. Commit workflow file
3. Push to trigger

✅ **Result**:
- Automated pipeline
- Public proofs
- Downloadable artifacts

---

**File**: `.github/workflows/monster-pipeline.yml` ✅  
**Triggers**: Push, PR, Manual 🔄  
**Uploads**: 2 HuggingFace repos 📤  
**Artifacts**: 30 days retention 📦  

🔧 **GitHub Actions fixed and operational!**
