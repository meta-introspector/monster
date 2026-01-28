#!/usr/bin/env bash

SEEDS=(
    "🌙 binary moon"
    "🌊 wave crest"  
    "⭐ star prime"
    "🎭 mask symmetry"
    "🎪 circus tent"
    "red crimson scarlet"
    "blue azure cobalt"
    "green emerald jade"
    "mathematician Conway"
    "mathematician Griess"
    "mathematician Fischer"
    "mathematician Thompson"
    "mathematician Leech"
    "group theorist Conway"
    "algebraist Griess"
    "professor Fischer"
)

for seed in "${SEEDS[@]}"; do
    echo "=== Seed: $seed ==="
    ./trace_regs.sh "$seed" 2>&1 | tail -5
    echo ""
done
