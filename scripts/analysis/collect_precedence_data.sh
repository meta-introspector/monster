#!/bin/bash
# Collect precedence data from all available proof assistants

OUTPUT="precedence_data.txt"
> "$OUTPUT"

echo "=== CROSS-SYSTEM PRECEDENCE ANALYSIS ===" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

# 1. Spectral (Lean2) - We have this
echo "1. SPECTRAL (Lean2)" | tee -a "$OUTPUT"
if [ -d spectral ]; then
    echo "Precedence levels found:" | tee -a "$OUTPUT"
    grep -rh "infixl\|infixr" spectral --include="*.hlean" | \
        grep -v "^--" | \
        sed 's/.*`\([^`]*\)`:\([0-9]\+\).*/\2: \1/' | \
        sort -n | uniq | tee -a "$OUTPUT"
    echo "" | tee -a "$OUTPUT"
    
    echo "Count by level:" | tee -a "$OUTPUT"
    grep -rh "infixl.*:[0-9]\+\|infixr.*:[0-9]\+" spectral --include="*.hlean" | \
        sed 's/.*:\([0-9]\+\).*/\1/' | sort -n | uniq -c | tee -a "$OUTPUT"
    echo "" | tee -a "$OUTPUT"
fi

# 2. Lean4 mathlib
echo "2. LEAN4 MATHLIB" | tee -a "$OUTPUT"
if [ -d .lake/packages/mathlib ]; then
    echo "Searching for precedence 71:" | tee -a "$OUTPUT"
    grep -rn ":71" .lake/packages/mathlib --include="*.lean" | head -10 | tee -a "$OUTPUT"
    echo "" | tee -a "$OUTPUT"
    
    echo "Common precedence levels:" | tee -a "$OUTPUT"
    grep -rh "infixl.*:[0-9]\+\|infixr.*:[0-9]\+" .lake/packages/mathlib --include="*.lean" | \
        sed 's/.*:\([0-9]\+\).*/\1/' | sort -n | uniq -c | sort -rn | head -20 | tee -a "$OUTPUT"
    echo "" | tee -a "$OUTPUT"
fi

# 3. Coq (if available)
echo "3. COQ STDLIB" | tee -a "$OUTPUT"
if [ -d ~/.opam/default/lib/coq ]; then
    echo "Searching for level 71:" | tee -a "$OUTPUT"
    grep -rn "at level 71" ~/.opam/default/lib/coq --include="*.v" | head -10 | tee -a "$OUTPUT"
    echo "" | tee -a "$OUTPUT"
    
    echo "Common levels:" | tee -a "$OUTPUT"
    grep -rh "at level [0-9]\+" ~/.opam/default/lib/coq --include="*.v" | \
        sed 's/.*at level \([0-9]\+\).*/\1/' | sort -n | uniq -c | sort -rn | head -20 | tee -a "$OUTPUT"
    echo "" | tee -a "$OUTPUT"
else
    echo "Coq not found" | tee -a "$OUTPUT"
    echo "" | tee -a "$OUTPUT"
fi

# 4. Monster Prime Analysis
echo "4. MONSTER PRIME PRECEDENCE COUNTS" | tee -a "$OUTPUT"
echo "Prime | Spectral | Lean4 | Coq" | tee -a "$OUTPUT"
echo "------|----------|-------|-----" | tee -a "$OUTPUT"

for p in 2 3 5 7 11 13 17 19 23 29 31 41 47 59 71; do
    spectral_count=0
    lean4_count=0
    coq_count=0
    
    if [ -d spectral ]; then
        spectral_count=$(grep -r ":$p\b" spectral --include="*.hlean" 2>/dev/null | wc -l)
    fi
    
    if [ -d .lake/packages/mathlib ]; then
        lean4_count=$(grep -r ":$p\b" .lake/packages/mathlib --include="*.lean" 2>/dev/null | wc -l)
    fi
    
    if [ -d ~/.opam/default/lib/coq ]; then
        coq_count=$(grep -r "at level $p\b" ~/.opam/default/lib/coq --include="*.v" 2>/dev/null | wc -l)
    fi
    
    printf "%5d | %8d | %5d | %3d\n" $p $spectral_count $lean4_count $coq_count | tee -a "$OUTPUT"
done

echo "" | tee -a "$OUTPUT"
echo "=== SUMMARY ===" | tee -a "$OUTPUT"
echo "Data saved to: $OUTPUT"
echo ""
echo "Key findings:"
echo "- Spectral uses precedence 71 for graded multiplication"
echo "- Check if other systems use 71 or other Monster primes"
echo "- Compare precedence schemes across systems"
