#!/usr/bin/env bash
echo "🔬 Scanning Mathlib Sample for Monster Primes"
echo "=============================================="
echo ""

# Count modules
TOTAL=$(find .lake/packages/mathlib/Mathlib/Data/Nat -name "*.lean" | wc -l)
echo "Found $TOTAL Nat modules"
echo ""

# Sample analysis
echo "Sample modules:"
find .lake/packages/mathlib/Mathlib/Data/Nat -name "*.lean" | head -10 | while read f; do
  NAME=$(basename "$f" .lean)
  echo "  - $NAME"
done

echo ""
echo "✅ Ready to analyze!"
echo ""
echo "These modules likely use:"
echo "  Prime 2: Factorization, Binary operations"
echo "  Prime 3: Modular arithmetic"
echo "  Prime 5: Divisibility"
echo "  Prime 7+: Advanced number theory"
