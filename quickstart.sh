#!/bin/bash
# Quick start for literate programming workflow

echo "🚀 MONSTER LITERATE PROGRAMMING - QUICK START"
echo "=============================================="
echo

# Step 1: Enter Nix environment
echo "Step 1: Entering Nix environment..."
nix develop --command bash << 'EOF'

echo "✅ Nix environment loaded"
echo

# Step 2: Build everything
echo "Step 2: Building literate proof..."
make workflow

echo
echo "=============================================="
echo "✅ BUILD COMPLETE"
echo "=============================================="
echo

# Check what was created
if [ -f monster_proof.pdf ]; then
    echo "📄 monster_proof.pdf - $(du -h monster_proof.pdf | cut -f1)"
fi

if [ -f monster_proof.rs ]; then
    echo "🦀 monster_proof.rs - $(wc -l < monster_proof.rs) lines"
fi

PNG_COUNT=$(ls monster_proof_page-*.png 2>/dev/null | wc -l)
if [ $PNG_COUNT -gt 0 ]; then
    echo "🖼️  $PNG_COUNT PNG images created"
fi

echo
echo "Next: Review with vision model"
echo "  ./review_paper_with_vision.sh"

EOF
