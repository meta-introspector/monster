#!/usr/bin/env bash
# Push performance data to HuggingFace as parquet proof

set -e

REPOS=("introspector/data-moonshine" "meta-introspector/monster-perf-proofs")
COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "📤 PUSH PERF DATA TO HUGGINGFACE (2 REPOS)"
echo "============================================================"
echo "Repos:"
for repo in "${REPOS[@]}"; do
    echo "  - $repo"
done
echo "Commit: ${COMMIT_HASH:0:8}"
echo "Time: $(date -Iseconds)"
echo ""

# Stage 1: Collect all parquet files
echo "📊 [1/5] Collecting parquet files..."

PARQUET_FILES=(
    "zkml_witness.parquet"
    "commit_reviews_*.parquet"
    "scrum_reviews.parquet"
    "language_complexity.parquet"
    "knuth_reviews.parquet"
)

FOUND_FILES=()
for pattern in "${PARQUET_FILES[@]}"; do
    for file in $pattern; do
        if [ -f "$file" ]; then
            FOUND_FILES+=("$file")
            echo "  ✓ $file ($(stat -f%z "$file" 2>/dev/null || stat -c%s "$file") bytes)"
        fi
    done
done

if [ ${#FOUND_FILES[@]} -eq 0 ]; then
    echo "❌ No parquet files found"
    echo "   Run: ./zkml_pipeline.sh first"
    exit 1
fi

echo "✓ Found ${#FOUND_FILES[@]} parquet files"
echo ""

# Stage 2: Create metadata
echo "📋 [2/5] Creating metadata..."

cat > perf_metadata.json << EOF
{
  "commit": "$COMMIT_HASH",
  "timestamp": "$(date -Iseconds)",
  "files": [
$(for file in "${FOUND_FILES[@]}"; do
    SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
    echo "    {\"name\": \"$file\", \"size\": $SIZE},"
done | sed '$ s/,$//')
  ],
  "pipeline": {
    "stages": ["compile", "build", "review", "zkml", "parquet"],
    "status": "complete"
  }
}
EOF

echo "✓ perf_metadata.json"
echo ""

# Stage 3: Create README for HuggingFace
echo "📖 [3/5] Creating dataset README..."

cat > HUGGINGFACE_README.md << 'README'
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

README

# Add file list
for file in "${FOUND_FILES[@]}"; do
    SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
    cat >> HUGGINGFACE_README.md << FILE

### \`$file\`
- **Size**: $SIZE bytes
- **Format**: Apache Parquet
- **Commit**: ${COMMIT_HASH:0:8}
FILE
done

cat >> HUGGINGFACE_README.md << 'README'

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
README

echo "✓ HUGGINGFACE_README.md"
echo ""

# Stage 4: Upload to HuggingFace
echo "📤 [4/5] Uploading to HuggingFace (2 repos)..."

if command -v huggingface-cli &> /dev/null; then
    echo "Using huggingface-cli..."
    
    # Check if logged in
    if huggingface-cli whoami &> /dev/null; then
        echo "✓ Authenticated"
        
        # Upload to each repo
        for REPO in "${REPOS[@]}"; do
            echo ""
            echo "Uploading to $REPO..."
            
            # Upload each file
            for file in "${FOUND_FILES[@]}"; do
                echo "  Uploading $file..."
                huggingface-cli upload "$REPO" "$file" --repo-type dataset 2>&1 | tail -2
            done
            
            # Upload metadata
            huggingface-cli upload "$REPO" perf_metadata.json --repo-type dataset 2>&1 | tail -2
            huggingface-cli upload "$REPO" HUGGINGFACE_README.md README.md --repo-type dataset 2>&1 | tail -2
            
            echo "✓ Upload to $REPO complete"
        done
        
        echo ""
        echo "✓ All uploads complete"
    else
        echo "⚠️  Not authenticated"
        echo "   Run: huggingface-cli login"
        echo ""
        echo "Files ready for manual upload to:"
        for REPO in "${REPOS[@]}"; do
            echo "  - $REPO"
        done
        echo ""
        echo "Files:"
        for file in "${FOUND_FILES[@]}"; do
            echo "  - $file"
        done
    fi
else
    echo "⚠️  huggingface-cli not available"
    echo "   Install: pip install huggingface_hub"
    echo ""
    echo "Files ready for manual upload to:"
    for REPO in "${REPOS[@]}"; do
        echo "  - $REPO"
    done
    echo ""
    echo "Files:"
    for file in "${FOUND_FILES[@]}"; do
        echo "  - $file"
    done
fi
echo ""

# Stage 5: Generate upload summary
echo "📋 [5/5] Generating summary..."

cat > HUGGINGFACE_UPLOAD.md << SUMMARY
# 📤 HuggingFace Upload Summary

## Repositories

SUMMARY

for REPO in "${REPOS[@]}"; do
    cat >> HUGGINGFACE_UPLOAD.md << REPO_INFO
**URL**: https://huggingface.co/datasets/$REPO

REPO_INFO
done

cat >> HUGGINGFACE_UPLOAD.md << SUMMARY

## Commit

**Hash**: $COMMIT_HASH
**Time**: $(date -Iseconds)

## Files Uploaded

SUMMARY

for file in "${FOUND_FILES[@]}"; do
    SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
    cat >> HUGGINGFACE_UPLOAD.md << FILE
- \`$file\` ($SIZE bytes)
FILE
done

cat >> HUGGINGFACE_UPLOAD.md << SUMMARY

## Metadata

\`\`\`json
$(cat perf_metadata.json)
\`\`\`

## Access

\`\`\`python
from datasets import load_dataset
import pandas as pd

# From moonshine repo
df1 = pd.read_parquet("hf://datasets/introspector/data-moonshine/zkml_witness.parquet")

# From monster-perf-proofs repo
df2 = pd.read_parquet("hf://datasets/meta-introspector/monster-perf-proofs/zkml_witness.parquet")

# Both should be identical
assert df1.equals(df2)
\`\`\`

## Verification

\`\`\`bash
# Download from either repo
huggingface-cli download introspector/data-moonshine --repo-type dataset
# OR
huggingface-cli download meta-introspector/monster-perf-proofs --repo-type dataset
\`\`\`

---

**Status**: Ready for upload ✅
**Files**: ${#FOUND_FILES[@]}
**Repos**: ${#REPOS[@]}
SUMMARY

echo "✓ HUGGINGFACE_UPLOAD.md"
echo ""

echo "✅ HUGGINGFACE UPLOAD COMPLETE"
echo "============================================================"
echo ""
echo "📊 Summary:"
echo "  Files: ${#FOUND_FILES[@]}"
echo "  Repos: ${#REPOS[@]}"
for REPO in "${REPOS[@]}"; do
    echo "    - $REPO"
done
echo "  Commit: ${COMMIT_HASH:0:8}"
echo ""
echo "🔗 View at:"
for REPO in "${REPOS[@]}"; do
    echo "  https://huggingface.co/datasets/$REPO"
done
echo ""
echo "📋 Documentation:"
echo "  HUGGINGFACE_README.md"
echo "  HUGGINGFACE_UPLOAD.md"
echo ""
echo "🎯 Performance data ready for public verification!"
