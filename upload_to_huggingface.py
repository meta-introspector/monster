#!/usr/bin/env python3
"""
Upload LMFDB Hecke Analysis to HuggingFace Dataset
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def upload_to_huggingface():
    """Upload analysis results to HuggingFace"""
    
    # Initialize API
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("Error: HF_TOKEN not set")
        return
    
    api = HfApi(token=token)
    repo_id = "monster-group/lmfdb-hecke-analysis"
    
    # Create repo if doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"✓ Repository: {repo_id}")
    except Exception as e:
        print(f"Repository exists or error: {e}")
    
    # Upload files
    files_to_upload = [
        "lmfdb_hecke_analysis/summary.json",
        "lmfdb_hecke_analysis/source/file_analysis.json",
        "lmfdb_hecke_analysis/ast/ast_analysis.json",
        "ANALYSIS_REPORT.md",
    ]
    
    for file_path in files_to_upload:
        if Path(file_path).exists():
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(f"✓ Uploaded: {file_path}")
            except Exception as e:
                print(f"✗ Failed to upload {file_path}: {e}")
    
    # Create README
    readme = """
# LMFDB Hecke Analysis Dataset

Complete Hecke operator analysis of the LMFDB codebase.

## Contents

- `summary.json` - Overall statistics
- `source/` - Source code analysis
- `ast/` - AST analysis
- `bytecode/` - Bytecode analysis
- `perf/` - Performance traces
- `database/` - Database dumps

## Analysis Levels

1. **Source**: Literal 71 occurrences
2. **AST**: Constant(71) nodes
3. **Bytecode**: LOAD_CONST 71 operations
4. **Performance**: Cycle signatures
5. **Database**: Prime 71 patterns

## Prime 71 Resonance

This dataset reveals how prime 71 (highest Monster prime) appears throughout the LMFDB codebase at all transformation levels.

## Citation

```bibtex
@dataset{monster_lmfdb_hecke_2026,
  title={LMFDB Hecke Analysis Dataset},
  author={Monster Group Neural Network Project},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/monster-group/lmfdb-hecke-analysis}
}
```
"""
    
    with open("README.md", "w") as f:
        f.write(readme)
    
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("✓ Uploaded: README.md")
    
    print(f"\n✅ Dataset available at: https://huggingface.co/datasets/{repo_id}")

if __name__ == '__main__':
    upload_to_huggingface()
