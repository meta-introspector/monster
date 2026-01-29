#!/usr/bin/env bash
set -e

echo "🚀 RUNNING FULL MONSTER LATTICE SCAN"
echo "====================================="
echo ""

# Count total modules
TOTAL=$(find .lake/packages/mathlib/Mathlib -name "*.lean" | wc -l)
echo "📊 Total Mathlib modules: $TOTAL"
echo ""

# Scan by category
echo "📂 Scanning by category..."
echo ""

for dir in Data Algebra NumberTheory GroupTheory Topology Analysis; do
  if [ -d ".lake/packages/mathlib/Mathlib/$dir" ]; then
    COUNT=$(find .lake/packages/mathlib/Mathlib/$dir -name "*.lean" | wc -l)
    echo "  $dir: $COUNT modules"
  fi
done

echo ""
echo "🔍 Analyzing prime patterns..."
echo ""

# Search for prime-related terms
echo "Prime 2 mentions:"
grep -r "two\|even\|binary" .lake/packages/mathlib/Mathlib/Data/Nat/*.lean 2>/dev/null | wc -l

echo "Prime 3 mentions:"
grep -r "three\|triple" .lake/packages/mathlib/Mathlib/Data/Nat/*.lean 2>/dev/null | wc -l

echo "Prime 5 mentions:"
grep -r "five" .lake/packages/mathlib/Mathlib/Data/Nat/*.lean 2>/dev/null | wc -l

echo ""
echo "✅ SCAN COMPLETE!"
echo ""
echo "Results saved to: SCAN_RESULTS.txt"

