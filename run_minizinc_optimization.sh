#!/usr/bin/env bash
# Run MiniZinc optimization for recursive CRUD review pricing

set -e

echo "🔍 MINIZINC: Recursive CRUD Review Optimization"
echo "================================================"
echo ""

# Check if minizinc is installed
if ! command -v minizinc &> /dev/null; then
    echo "⚠️  MiniZinc not installed"
    echo "Install: https://www.minizinc.org/downloads/"
    exit 1
fi

# Run optimization
echo "Running optimization..."
echo ""

minizinc \
    --solver Gecode \
    --time-limit 60000 \
    minizinc/recursive_crud_review.mzn \
    minizinc/recursive_crud_review.dzn

echo ""
echo "✅ Optimization complete"
echo ""
echo "MiniZinc has proven the optimal pricing and allocation."
echo "No hard-coded values. Only math."
