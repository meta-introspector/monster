#!/usr/bin/env bash
set -e

echo "🎵 Cloning Harmonic Analysis Repos"
echo "=================================="
echo ""

repos=(
    "https://github.com/mathematical-tours/mathematical-tours.github.io"
    "https://github.com/JuliaMolSim/DFTK.jl"
    "https://github.com/JuliaApproximation/ApproxFun.jl"
)

for repo in "${repos[@]}"; do
    name=$(basename "$repo" .git)
    if [ -d "$name" ]; then
        echo "✓ $name already cloned"
    else
        echo "📥 Cloning $name..."
        git clone --depth 1 "$repo" 2>&1 | grep -E "Cloning|done" || true
    fi
done

echo ""
echo "✅ Done! Cloned $(ls -d */ 2>/dev/null | wc -l) repos"
