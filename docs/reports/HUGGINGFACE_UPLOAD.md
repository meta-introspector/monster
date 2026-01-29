# 📤 HuggingFace Upload Summary

## Repositories

**URL**: https://huggingface.co/datasets/introspector/data-moonshine

**URL**: https://huggingface.co/datasets/meta-introspector/monster-perf-proofs


## Commit

**Hash**: bdd8dd5fe81e8ac4e8b2913876d921cb61caa7c2
**Time**: 2026-01-29T06:09:35-05:00

## Files Uploaded

- `zkml_witness.parquet` (6049 bytes)

## Metadata

```json
{
  "commit": "bdd8dd5fe81e8ac4e8b2913876d921cb61caa7c2",
  "timestamp": "2026-01-29T06:09:35-05:00",
  "files": [
    {"name": "zkml_witness.parquet", "size": 6049}
  ],
  "pipeline": {
    "stages": ["compile", "build", "review", "zkml", "parquet"],
    "status": "complete"
  }
}
```

## Access

```python
from datasets import load_dataset
import pandas as pd

# From moonshine repo
df1 = pd.read_parquet("hf://datasets/introspector/data-moonshine/zkml_witness.parquet")

# From monster-perf-proofs repo
df2 = pd.read_parquet("hf://datasets/meta-introspector/monster-perf-proofs/zkml_witness.parquet")

# Both should be identical
assert df1.equals(df2)
```

## Verification

```bash
# Download from either repo
huggingface-cli download introspector/data-moonshine --repo-type dataset
# OR
huggingface-cli download meta-introspector/monster-perf-proofs --repo-type dataset
```

---

**Status**: Ready for upload ✅
**Files**: 1
**Repos**: 2
