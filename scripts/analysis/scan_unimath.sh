#!/usr/bin/env bash

UNIMATH_PATH="$HOME/experiments/UniMath"

echo "🔬 SCANNING UNIMATH FOR MONSTER STRUCTURE"
echo "=========================================="
echo ""

if [ ! -d "$UNIMATH_PATH" ]; then
    echo "❌ UniMath not found at $UNIMATH_PATH"
    exit 1
fi

cd "$UNIMATH_PATH"

echo "📊 BASIC STATISTICS:"
echo "-------------------"
total_v=$(find . -name "*.v" | wc -l)
echo "Total .v files: $total_v"

echo ""
echo "🎯 SCANNING FOR MONSTER PRIMES:"
echo "-------------------------------"

for prime in 2 3 5 7 11 13 17 19 23 29 31 41 47 59 71; do
    count=$(grep -r "\b${prime}\b" . --include="*.v" 2>/dev/null | wc -l)
    if [ $count -gt 0 ]; then
        printf "Prime %3d: %6d mentions\n" $prime $count
    fi
done

echo ""
echo "👹 SEARCHING FOR PRIME 71:"
echo "-------------------------"
grep -r "\b71\b" . --include="*.v" -l 2>/dev/null | head -5

echo ""
echo "🌳 ANALYZING TREE DEPTH:"
echo "-----------------------"
echo "Looking for deeply nested structures..."

# Find files with deep nesting (many parentheses)
echo "Files with deepest nesting:"
for file in $(find . -name "*.v" | head -100); do
    depth=$(grep -o '(' "$file" | wc -l)
    if [ $depth -gt 100 ]; then
        echo "  $file: ~$depth levels"
    fi
done | sort -t: -k2 -nr | head -5

echo ""
echo "🔍 SEARCHING FOR 'total2' (fiber bundle):"
echo "-----------------------------------------"
grep -r "total2" . --include="*.v" -c 2>/dev/null | sort -t: -k2 -nr | head -5

echo ""
echo "✅ Scan complete!"
