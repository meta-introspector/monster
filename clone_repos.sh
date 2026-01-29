#!/usr/bin/env bash
set -e

cd gap_sage_repos

echo "🔄 Cloning Monster/Moonshine repos"
echo "===================================="
echo ""

# Math/Group Theory repos only
repos=(
    "https://github.com/ChrisJefferson/BacktrackKit"
    "https://github.com/ThGroth/gap-equations"
    "https://github.com/fingolfin/BacktrackKit"
    "https://github.com/peal/BacktrackKit"
    "https://github.com/wucas/BacktrackKit"
    "https://github.com/brandonrayhaun/moonshine-code"
    "https://github.com/rafafrdz/braids-and-cryptography"
    "https://github.com/shreevatsa/misc-math"
)

for repo in "${repos[@]}"; do
    name=$(basename "$repo")
    if [ -d "$name" ]; then
        echo "✓ $name already cloned"
    else
        echo "📥 Cloning $name..."
        git clone --depth 1 "$repo" 2>&1 | grep -E "Cloning|done" || true
    fi
done

echo ""
echo "✅ Done! Cloned $(ls -d */ 2>/dev/null | wc -l) repos"
