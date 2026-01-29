---
license: mit
task_categories:
- other
tags:
- performance
- zero-knowledge
- formal-verification
- monster-group
size_categories:
- n<1K
---

# Monster Group Performance Proofs

Performance data and ZK-ML proofs from the Monster Group project.

## Dataset Description

This dataset contains:
- **Performance traces** from compilation and build
- **Review scores** from 9-persona review team
- **ZK-ML witnesses** proving constraint satisfaction
- **Language complexity** measurements (Coq, Lean4, Rust, Nix)

## Files


### `zkml_witness.parquet`
- **Size**: 6049 bytes
- **Format**: Apache Parquet
- **Commit**: bdd8dd5f

## Schema

### zkml_witness.parquet
```
commit_hash: string
timestamp: int64
compile_time_ms: int64
build_time_ms: int64
review_score: int64
cpu_cycles: int64
memory_peak_mb: int64
```

### commit_reviews_*.parquet
```
commit: string
timestamp: datetime
reviewer: string
comment: string
approved: boolean
cpu_cycles: int64
instructions: int64
```

### language_complexity.parquet
```
language: string
expr_depth: int32
type_depth: int32
func_nesting: int32
universe_level: int32
layer: int32
layer_name: string
monster_exponent: string
```

## Usage

```python
import pandas as pd

# Load ZK-ML witness
df = pd.read_parquet('zkml_witness.parquet')
print(df)

# Load reviews
reviews = pd.read_parquet('commit_reviews_*.parquet')
print(reviews.groupby('reviewer')['approved'].mean())

# Load complexity
complexity = pd.read_parquet('language_complexity.parquet')
print(complexity[complexity['layer'] == 7])
```

## Zero-Knowledge Proofs

Each commit includes a ZK-ML proof that verifies:
- ✅ Compile time < 5 minutes
- ✅ Build time < 10 minutes
- ✅ Review score >= 70/90
- ✅ CPU cycles < 10 billion
- ✅ Memory < 16 GB

**Without revealing** actual performance details.

## Citation

```bibtex
@dataset{monster_perf_proofs,
  title={Monster Group Performance Proofs},
  author={Meta-Introspector Project},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/meta-introspector/monster-perf-proofs}
}
```

## License

MIT License - See LICENSE file
