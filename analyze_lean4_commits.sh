#!/usr/bin/env bash
# Analyze Lean4 submodule commits by Monster primes

set -e

echo "🔍 Lean4 Submodule Commit Analysis by Monster Primes"
echo "===================================================="
echo ""

# Monster primes
PRIMES=(2 3 5 7 11 13 17 19 23 29 31 41 47 59 71)

# Output directory
OUTPUT_DIR="datasets/lean4_commit_analysis"
mkdir -p "$OUTPUT_DIR"

# Lean4 packages to analyze
PACKAGES=(
    "mathlib"
    "FLT"
    "Carleson"
    "batteries"
    "aesop"
    "proofwidgets"
)

echo "Packages to analyze: ${PACKAGES[@]}"
echo ""

# Analyze each package
for pkg in "${PACKAGES[@]}"; do
    PKG_DIR=".lake/packages/$pkg"
    
    if [ ! -d "$PKG_DIR" ]; then
        echo "⚠️  Package not found: $pkg"
        continue
    fi
    
    echo "📦 Analyzing: $pkg"
    
    # Get all commits
    cd "$PKG_DIR"
    TOTAL_COMMITS=$(git log --all --oneline | wc -l)
    echo "  Total commits: $TOTAL_COMMITS"
    
    # Get all authors
    AUTHORS=$(git log --all --format="%an" | sort -u)
    TOTAL_AUTHORS=$(echo "$AUTHORS" | wc -l)
    echo "  Total authors: $TOTAL_AUTHORS"
    
    # Analyze by prime
    for prime in "${PRIMES[@]}"; do
        # Count commits where hash mod prime == 0
        RESONANT_COMMITS=$(git log --all --pretty=format:"%H" | \
            awk -v p=$prime '{
                # Convert first 8 chars of hash to number
                hash = substr($1, 1, 8)
                num = 0
                for (i = 1; i <= length(hash); i++) {
                    c = substr(hash, i, 1)
                    if (c ~ /[0-9]/) num = num * 16 + c
                    else if (c == "a") num = num * 16 + 10
                    else if (c == "b") num = num * 16 + 11
                    else if (c == "c") num = num * 16 + 12
                    else if (c == "d") num = num * 16 + 13
                    else if (c == "e") num = num * 16 + 14
                    else if (c == "f") num = num * 16 + 15
                }
                if (num % p == 0) print $1
            }' | wc -l)
        
        PERCENTAGE=$(awk "BEGIN {printf \"%.2f\", ($RESONANT_COMMITS / $TOTAL_COMMITS) * 100}")
        echo "    Prime $prime: $RESONANT_COMMITS commits ($PERCENTAGE%)"
    done
    
    # Save detailed analysis
    cd - > /dev/null
    OUTPUT_FILE="$OUTPUT_DIR/${pkg}_analysis.json"
    cat > "$OUTPUT_FILE" << EOF
{
  "package": "$pkg",
  "total_commits": $TOTAL_COMMITS,
  "total_authors": $TOTAL_AUTHORS,
  "prime_resonance": {
EOF
    
    for i in "${!PRIMES[@]}"; do
        prime="${PRIMES[$i]}"
        RESONANT=$(git log --all --pretty=format:"%H" | \
            awk -v p=$prime '{
                hash = substr($1, 1, 8)
                num = 0
                for (i = 1; i <= length(hash); i++) {
                    c = substr(hash, i, 1)
                    if (c ~ /[0-9]/) num = num * 16 + c
                    else if (c == "a") num = num * 16 + 10
                    else if (c == "b") num = num * 16 + 11
                    else if (c == "c") num = num * 16 + 12
                    else if (c == "d") num = num * 16 + 13
                    else if (c == "e") num = num * 16 + 14
                    else if (c == "f") num = num * 16 + 15
                }
                if (num % p == 0) print $1
            }' | wc -l)
        
        if [ $i -lt $((${#PRIMES[@]} - 1)) ]; then
            echo "    \"$prime\": $RESONANT," >> "$OUTPUT_FILE"
        else
            echo "    \"$prime\": $RESONANT" >> "$OUTPUT_FILE"
        fi
    done
    
    cat >> "$OUTPUT_FILE" << EOF
  },
  "authors": [
EOF
    
    # Top 10 authors by commit count
    git log --all --format="%an" | sort | uniq -c | sort -rn | head -10 | \
        awk '{
            count = $1
            $1 = ""
            author = substr($0, 2)
            gsub(/"/, "\\\"", author)
            print "    {\"author\": \"" author "\", \"commits\": " count "}"
        }' | paste -sd ',' >> "$OUTPUT_FILE"
    
    cat >> "$OUTPUT_FILE" << EOF

  ]
}
EOF
    
    cd - > /dev/null
    echo "  ✓ Saved: $OUTPUT_FILE"
    echo ""
done

# Generate summary
SUMMARY_FILE="$OUTPUT_DIR/summary.json"
echo "📊 Generating summary..."

cat > "$SUMMARY_FILE" << EOF
{
  "analysis_timestamp": "$(date -Iseconds)",
  "packages_analyzed": [
EOF

for i in "${!PACKAGES[@]}"; do
    pkg="${PACKAGES[$i]}"
    if [ -f "$OUTPUT_DIR/${pkg}_analysis.json" ]; then
        if [ $i -lt $((${#PACKAGES[@]} - 1)) ]; then
            echo "    \"$pkg\"," >> "$SUMMARY_FILE"
        else
            echo "    \"$pkg\"" >> "$SUMMARY_FILE"
        fi
    fi
done

cat >> "$SUMMARY_FILE" << EOF
  ],
  "monster_primes": [${PRIMES[@]}]
}
EOF

echo "✓ Summary: $SUMMARY_FILE"
echo ""
echo "✅ Analysis complete"
echo ""
echo "Results in: $OUTPUT_DIR/"
echo "  - ${PACKAGES[0]}_analysis.json"
echo "  - ${PACKAGES[1]}_analysis.json"
echo "  - ..."
echo "  - summary.json"
