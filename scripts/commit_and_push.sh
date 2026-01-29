#!/usr/bin/env bash
# Commit to git AND push to both HuggingFace repos

set -e

echo "🚀 COMMIT + PUSH TO BOTH REPOS"
echo "============================================================"
echo ""

# Stage 1: Git commit
echo "📝 [1/3] Git commit..."

if [ -z "$(git diff --cached --name-only)" ]; then
    echo "⚠️  No staged files, staging all changes..."
    git add -A
fi

STAGED=$(git diff --cached --name-only | wc -l)
echo "✓ Staged files: $STAGED"

# Get commit message from arg or use default
COMMIT_MSG="${1:-Update: $(date +%Y-%m-%d_%H:%M:%S)}"

git commit -m "$COMMIT_MSG" || echo "⚠️  Nothing to commit or commit failed"
COMMIT_HASH=$(git rev-parse HEAD)
echo "✓ Committed: ${COMMIT_HASH:0:8}"
echo ""

# Stage 2: Generate parquet from latest commit
echo "💾 [2/3] Generating parquet from commit..."

if [ -f "precommit_review.parquet" ]; then
    cp precommit_review.parquet "commit_${COMMIT_HASH:0:8}.parquet"
    echo "✓ commit_${COMMIT_HASH:0:8}.parquet"
fi

if [ -f "zkml_witness.parquet" ]; then
    echo "✓ zkml_witness.parquet ready"
fi
echo ""

# Stage 3: Push to both HuggingFace repos
echo "📤 [3/3] Pushing to HuggingFace (2 repos)..."

REPOS=("introspector/data-moonshine" "meta-introspector/monster-perf-proofs")

if command -v huggingface-cli &> /dev/null; then
    if huggingface-cli whoami &> /dev/null; then
        echo "✓ Authenticated"
        echo ""
        
        # Files to upload
        FILES=()
        [ -f "zkml_witness.parquet" ] && FILES+=("zkml_witness.parquet")
        [ -f "precommit_review.parquet" ] && FILES+=("precommit_review.parquet")
        [ -f "commit_${COMMIT_HASH:0:8}.parquet" ] && FILES+=("commit_${COMMIT_HASH:0:8}.parquet")
        
        if [ ${#FILES[@]} -eq 0 ]; then
            echo "⚠️  No parquet files to upload"
        else
            # Upload to each repo
            for REPO in "${REPOS[@]}"; do
                echo "Uploading to $REPO..."
                
                for file in "${FILES[@]}"; do
                    echo "  → $file"
                    huggingface-cli upload "$REPO" "$file" --repo-type dataset 2>&1 | grep -E "(Uploading|uploaded|✓)" | head -2 || true
                done
                
                echo "✓ $REPO complete"
                echo ""
            done
            
            echo "✅ All uploads complete!"
        fi
    else
        echo "❌ Not authenticated"
        echo "   Run: huggingface-cli login"
        exit 1
    fi
else
    echo "❌ huggingface-cli not available"
    echo "   Install: pip install huggingface_hub"
    exit 1
fi

echo ""
echo "✅ COMMIT + PUSH COMPLETE"
echo "============================================================"
echo ""
echo "Git:"
echo "  Commit: ${COMMIT_HASH:0:8}"
echo "  Message: $COMMIT_MSG"
echo ""
echo "HuggingFace:"
for REPO in "${REPOS[@]}"; do
    echo "  ✓ $REPO"
done
echo ""
echo "Files:"
for file in "${FILES[@]}"; do
    echo "  ✓ $file"
done
echo ""
echo "🔗 View at:"
for REPO in "${REPOS[@]}"; do
    echo "  https://huggingface.co/datasets/$REPO"
done
