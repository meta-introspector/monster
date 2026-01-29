#!/usr/bin/env bash

DOCS=(
    "RESULTS.md"
    "EXPERIMENT_SUMMARY.md"
    "../iarelife/RESIDUE_REPORT.md"
    "../../MATHEMATICAL_PROOF.md"
)

echo "📚 Feeding documents to model for verification"
echo "=============================================="
echo ""

for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        echo "=== Processing: $doc ==="
        
        # Extract key findings (first 500 chars)
        CONTENT=$(head -c 500 "$doc")
        
        PROMPT="Document: $doc

$CONTENT

As mathematician Conway, verify this claim: LLM registers show 80% divisibility by prime 2, 49% by prime 3, matching error correction codes. Valid?"
        
        ./trace_regs.sh "$PROMPT"
        echo ""
    fi
done

echo "✓ All documents processed"
cargo run --release --bin view-logs
