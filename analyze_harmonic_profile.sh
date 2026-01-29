#!/usr/bin/env bash
# Harmonic profile and eigenform analysis of Lean4 commits over time

set -e

echo "🎵 Harmonic Profile & Eigenform Analysis"
echo "========================================"
echo ""

PKG_DIR=".lake/packages/mathlib"
OUTPUT_DIR="datasets/lean4_harmonic_analysis"
mkdir -p "$OUTPUT_DIR"

cd "$PKG_DIR"

echo "📊 Extracting commit data..."

# Extract all commits with author, timestamp, and hash
git log --all --pretty=format:"%H|%an|%at" > /tmp/mathlib_commits.txt

TOTAL=$(wc -l < /tmp/mathlib_commits.txt)
echo "Total commits: $TOTAL"
echo ""

# Monster primes
PRIMES=(2 3 5 7 11 13 17 19 23 29 31 41 47 59 71)

echo "🎵 Calculating harmonic profile by author..."

# Get top 20 authors
TOP_AUTHORS=$(git log --all --format="%an" | sort | uniq -c | sort -rn | head -20 | awk '{$1=""; print substr($0,2)}')

# Analyze each author
while IFS= read -r author; do
    echo "  Author: $author"
    
    # Get author's commits
    AUTHOR_COMMITS=$(git log --all --author="$author" --pretty=format:"%H|%at")
    AUTHOR_COUNT=$(echo "$AUTHOR_COMMITS" | wc -l)
    
    # Calculate prime resonance for this author
    cd - > /dev/null
    AUTHOR_FILE="$OUTPUT_DIR/author_$(echo "$author" | tr ' ' '_' | tr -cd '[:alnum:]_').json"
    cd "$PKG_DIR"
    
    cat > "../../$AUTHOR_FILE" << EOF
{
  "author": "$author",
  "total_commits": $AUTHOR_COUNT,
  "prime_harmonics": {
EOF
    
    for i in "${!PRIMES[@]}"; do
        prime="${PRIMES[$i]}"
        
        # Count commits resonating with this prime
        RESONANT=$(echo "$AUTHOR_COMMITS" | awk -F'|' -v p=$prime '{
            hash = $1
            num = 0
            for (i = 1; i <= 8 && i <= length(hash); i++) {
                c = substr(hash, i, 1)
                if (c ~ /[0-9]/) num = num * 16 + c
                else if (c == "a") num = num * 16 + 10
                else if (c == "b") num = num * 16 + 11
                else if (c == "c") num = num * 16 + 12
                else if (c == "d") num = num * 16 + 13
                else if (c == "e") num = num * 16 + 14
                else if (c == "f") num = num * 16 + 15
            }
            if (num % p == 0) print
        }' | wc -l)
        
        FREQ=$(awk "BEGIN {printf \"%.4f\", $RESONANT / $AUTHOR_COUNT}")
        
        if [ $i -lt $((${#PRIMES[@]} - 1)) ]; then
            echo "    \"$prime\": {\"count\": $RESONANT, \"frequency\": $FREQ}," >> "../../$AUTHOR_FILE"
        else
            echo "    \"$prime\": {\"count\": $RESONANT, \"frequency\": $FREQ}" >> "../../$AUTHOR_FILE"
        fi
    done
    
    cat >> "../../$AUTHOR_FILE" << EOF
  }
}
EOF
    
done <<< "$TOP_AUTHORS"

cd - > /dev/null

echo ""
echo "📈 Calculating eigenform over time..."

# Split commits into time buckets (yearly)
EIGENFORM_FILE="$OUTPUT_DIR/eigenform_timeline.json"

cat > "$EIGENFORM_FILE" << 'EOF'
{
  "project": "mathlib",
  "eigenform_timeline": [
EOF

# Get first and last commit timestamps
cd "$PKG_DIR"
FIRST_TS=$(git log --all --reverse --pretty=format:"%at" | head -1)
LAST_TS=$(git log --all --pretty=format:"%at" | head -1)

FIRST_YEAR=$(date -d "@$FIRST_TS" +%Y)
LAST_YEAR=$(date -d "@$LAST_TS" +%Y)

echo "  Time range: $FIRST_YEAR - $LAST_YEAR"

# Analyze each year
for year in $(seq $FIRST_YEAR $LAST_YEAR); do
    START_TS=$(date -d "$year-01-01" +%s)
    END_TS=$(date -d "$((year+1))-01-01" +%s)
    
    echo "    Year $year..."
    
    # Get commits in this year
    YEAR_COMMITS=$(git log --all --since="$START_TS" --until="$END_TS" --pretty=format:"%H")
    YEAR_COUNT=$(echo "$YEAR_COMMITS" | grep -c . || echo 0)
    
    if [ "$YEAR_COUNT" -eq 0 ]; then
        continue
    fi
    
    # Calculate prime harmonics for this year
    cd - > /dev/null
    cat >> "$EIGENFORM_FILE" << EOF
    {
      "year": $year,
      "commits": $YEAR_COUNT,
      "harmonics": {
EOF
    cd "$PKG_DIR"
    
    for i in "${!PRIMES[@]}"; do
        prime="${PRIMES[$i]}"
        
        RESONANT=$(echo "$YEAR_COMMITS" | awk -v p=$prime '{
            hash = $1
            num = 0
            for (i = 1; i <= 8 && i <= length(hash); i++) {
                c = substr(hash, i, 1)
                if (c ~ /[0-9]/) num = num * 16 + c
                else if (c == "a") num = num * 16 + 10
                else if (c == "b") num = num * 16 + 11
                else if (c == "c") num = num * 16 + 12
                else if (c == "d") num = num * 16 + 13
                else if (c == "e") num = num * 16 + 14
                else if (c == "f") num = num * 16 + 15
            }
            if (num % p == 0) print
        }' | wc -l)
        
        cd - > /dev/null
        if [ $i -lt $((${#PRIMES[@]} - 1)) ]; then
            echo "        \"$prime\": $RESONANT," >> "$EIGENFORM_FILE"
        else
            echo "        \"$prime\": $RESONANT" >> "$EIGENFORM_FILE"
        fi
        cd "$PKG_DIR"
    done
    
    cd - > /dev/null
    if [ "$year" -lt "$LAST_YEAR" ]; then
        echo "      }" >> "$EIGENFORM_FILE"
        echo "    }," >> "$EIGENFORM_FILE"
    else
        echo "      }" >> "$EIGENFORM_FILE"
        echo "    }" >> "$EIGENFORM_FILE"
    fi
    cd "$PKG_DIR"
done

cd - > /dev/null

cat >> "$EIGENFORM_FILE" << 'EOF'
  ]
}
EOF

echo ""
echo "✅ Analysis complete"
echo ""
echo "Results:"
echo "  - Author harmonics: $OUTPUT_DIR/author_*.json"
echo "  - Eigenform timeline: $OUTPUT_DIR/eigenform_timeline.json"
echo ""
echo "🎵 Harmonic profile calculated"
echo "📈 Eigenform over time extracted"
