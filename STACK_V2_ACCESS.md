# ✅ The Stack v2 Access Confirmed!

**Date**: 2026-01-29  
**Status**: ✅ Authenticated

## Access Details

- **Dataset**: bigcode/the-stack-v2
- **Downloads**: 5,333
- **Likes**: 449
- **Your Token**: ~/.hf (38 bytes)

## Available Languages

The Stack v2 includes 600+ languages, including:
- Rust ✅
- Python
- JavaScript
- TypeScript
- Go
- Java
- C/C++
- And 590+ more...

## Current Issue

⚠️ **NumPy version conflict** prevents local streaming:
- System NumPy: 2.2.6
- Required: <2.0

## Solutions

### 1. Use GitHub Actions (Recommended)

The workflow `.github/workflows/stack-analysis.yml` will run in a clean environment:

```bash
gh workflow run "Analyze with The Stack v2"
```

### 2. Fix NumPy Locally

```bash
pip install 'numpy<2.0' --force-reinstall
python3 analyze_with_stack.py
```

### 3. Use HuggingFace Datasets Viewer

Visit: https://huggingface.co/datasets/bigcode/the-stack-v2/viewer

## Next Steps

1. **Trigger workflow** to run analysis in CI
2. **Download results** from artifacts
3. **Compare** Monster vs Stack v2 metrics
4. **Upload** to HuggingFace datasets

## Workflow Command

```bash
# Add HF_TOKEN to GitHub secrets first
gh secret set HF_TOKEN < ~/.hf

# Then trigger
gh workflow run "Analyze with The Stack v2"

# Check status
gh run list --workflow="Analyze with The Stack v2"
```

## Expected Output

Once workflow runs:
- `stack_v2_rust_sample.parquet` - 50-100 Rust samples
- `comparison_report.json` - Monster vs Stack metrics
- Uploaded to both HuggingFace repos

## Summary

✅ **Access confirmed**  
⚠️ **Local NumPy conflict**  
💡 **Use GitHub Actions**  
🚀 **Ready to benchmark!**
