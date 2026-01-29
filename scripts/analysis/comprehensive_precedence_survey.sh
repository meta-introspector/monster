#!/bin/bash
# Comprehensive Precedence Data Collection with Citations
# Collects ALL operator precedence levels from Coq, MetaCoq, UniMath, Lean4, and Spectral

set -e

OUTPUT_DIR="datasets/precedence_survey"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SURVEY_FILE="$OUTPUT_DIR/precedence_survey_${TIMESTAMP}.csv"
CITATIONS_FILE="$OUTPUT_DIR/citations_${TIMESTAMP}.md"

# Create citations file
touch "$CITATIONS_FILE"

echo "=== COMPREHENSIVE PRECEDENCE SURVEY ===" > "$CITATIONS_FILE"
echo "Timestamp: $(date)" >> "$CITATIONS_FILE"
echo "" >> "$CITATIONS_FILE"

# CSV Header
echo "system,project,file,line,operator,precedence,git_url,commit,branch" > "$SURVEY_FILE"

# Function to add citation
add_citation() {
    local system=$1
    local project=$2
    local git_url=$3
    local commit=$4
    local branch=$5
    
    echo "## $system - $project" >> "$CITATIONS_FILE"
    echo "- **Git URL**: $git_url" >> "$CITATIONS_FILE"
    echo "- **Commit**: $commit" >> "$CITATIONS_FILE"
    echo "- **Branch**: $branch" >> "$CITATIONS_FILE"
    echo "- **Scan Date**: $(date)" >> "$CITATIONS_FILE"
    echo "" >> "$CITATIONS_FILE"
}

# Function to extract and record precedence
extract_precedence() {
    local system=$1
    local project=$2
    local dir=$3
    local pattern=$4
    local git_url=$5
    
    if [ ! -d "$dir" ]; then
        echo "⚠️  $project not found at $dir" | tee -a "$CITATIONS_FILE"
        return
    fi
    
    cd "$dir"
    
    # Get git info
    local commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    local branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    
    add_citation "$system" "$project" "$git_url" "$commit" "$branch"
    
    echo "Scanning $project..." | tee -a "$CITATIONS_FILE"
    
    # Extract precedence data
    local count=0
    while IFS= read -r line; do
        local file=$(echo "$line" | cut -d: -f1)
        local linenum=$(echo "$line" | cut -d: -f2)
        local content=$(echo "$line" | cut -d: -f3-)
        
        # Extract operator and precedence based on pattern
        local operator=""
        local precedence=""
        
        if [[ "$pattern" == "lean" ]]; then
            operator=$(echo "$content" | grep -oP '(?<=infixl "|infixr ")[^"]+' || echo "")
            precedence=$(echo "$content" | grep -oP '(?<=:)\d+' || echo "")
        elif [[ "$pattern" == "coq" ]]; then
            operator=$(echo "$content" | grep -oP '(?<=")[^"]+(?=")' | head -1 || echo "")
            precedence=$(echo "$content" | grep -oP '(?<=at level )\d+' || echo "")
        fi
        
        if [ -n "$precedence" ]; then
            echo "$system,$project,$file,$linenum,$operator,$precedence,$git_url,$commit,$branch" >> "$SURVEY_FILE"
            ((count++))
        fi
    done < <(eval "$pattern")
    
    echo "Found $count precedence declarations" | tee -a "$CITATIONS_FILE"
    echo "" | tee -a "$CITATIONS_FILE"
    
    cd - > /dev/null
}

# 1. Spectral (Lean2)
echo "=== 1. SPECTRAL (Lean2) ===" | tee -a "$CITATIONS_FILE"
if [ -d "spectral" ]; then
    extract_precedence "Lean2" "Spectral" "spectral" \
        "grep -rn 'infixl\|infixr' --include='*.hlean'" \
        "https://github.com/cmu-phil/Spectral"
fi

# 2. Lean4 Mathlib
echo "=== 2. LEAN4 MATHLIB ===" | tee -a "$CITATIONS_FILE"
if [ -d ".lake/packages/mathlib" ]; then
    extract_precedence "Lean4" "Mathlib" ".lake/packages/mathlib" \
        "grep -rn 'infixl.*:[0-9]\+\|infixr.*:[0-9]\+' --include='*.lean'" \
        "https://github.com/leanprover-community/mathlib4"
fi

# 3. Coq Standard Library
echo "=== 3. COQ STDLIB ===" | tee -a "$CITATIONS_FILE"
if [ -d "$HOME/.opam/default/lib/coq" ]; then
    extract_precedence "Coq" "Stdlib" "$HOME/.opam/default/lib/coq" \
        "grep -rn 'at level [0-9]\+' --include='*.v'" \
        "https://github.com/coq/coq"
fi

# 4. Clone and scan UniMath if not present
echo "=== 4. UNIMATH ===" | tee -a "$CITATIONS_FILE"
if [ ! -d "external/UniMath" ]; then
    echo "Cloning UniMath..." | tee -a "$CITATIONS_FILE"
    mkdir -p external
    git clone --depth 1 https://github.com/UniMath/UniMath external/UniMath
fi
if [ -d "external/UniMath" ]; then
    extract_precedence "Coq" "UniMath" "external/UniMath" \
        "grep -rn 'at level [0-9]\+' --include='*.v'" \
        "https://github.com/UniMath/UniMath"
fi

# 5. Clone and scan MetaCoq if not present
echo "=== 5. METACOQ ===" | tee -a "$CITATIONS_FILE"
if [ ! -d "external/metacoq" ]; then
    echo "Cloning MetaCoq..." | tee -a "$CITATIONS_FILE"
    mkdir -p external
    git clone --depth 1 https://github.com/MetaCoq/metacoq external/metacoq
fi
if [ -d "external/metacoq" ]; then
    extract_precedence "Coq" "MetaCoq" "external/metacoq" \
        "grep -rn 'at level [0-9]\+' --include='*.v'" \
        "https://github.com/MetaCoq/metacoq"
fi

# Generate summary statistics
echo "=== SUMMARY STATISTICS ===" | tee -a "$CITATIONS_FILE"
echo "" | tee -a "$CITATIONS_FILE"

# Count by system
echo "Precedence declarations by system:" | tee -a "$CITATIONS_FILE"
tail -n +2 "$SURVEY_FILE" | cut -d, -f1 | sort | uniq -c | tee -a "$CITATIONS_FILE"
echo "" | tee -a "$CITATIONS_FILE"

# Count occurrences of 71
echo "Occurrences of precedence 71:" | tee -a "$CITATIONS_FILE"
grep ",71," "$SURVEY_FILE" | wc -l | tee -a "$CITATIONS_FILE"
echo "" | tee -a "$CITATIONS_FILE"

# List all uses of 71
echo "All uses of precedence 71:" | tee -a "$CITATIONS_FILE"
grep ",71," "$SURVEY_FILE" | tee -a "$CITATIONS_FILE"
echo "" | tee -a "$CITATIONS_FILE"

# Monster prime distribution
echo "Monster prime distribution:" | tee -a "$CITATIONS_FILE"
for p in 2 3 5 7 11 13 17 19 23 29 31 41 47 59 71; do
    count=$(grep ",$p," "$SURVEY_FILE" | wc -l)
    echo "Prime $p: $count occurrences" | tee -a "$CITATIONS_FILE"
done
echo "" | tee -a "$CITATIONS_FILE"

# Convert to Parquet
echo "Converting to Parquet..." | tee -a "$CITATIONS_FILE"
python3 << EOF
import pandas as pd
import sys

try:
    df = pd.read_csv("$SURVEY_FILE")
    parquet_file = "$OUTPUT_DIR/precedence_survey_${TIMESTAMP}.parquet"
    df.to_parquet(parquet_file, index=False)
    print(f"✓ Saved to {parquet_file}")
    
    # Print summary
    print(f"\nTotal records: {len(df)}")
    print(f"\nBy system:")
    print(df['system'].value_counts())
    print(f"\nPrecedence 71 occurrences: {len(df[df['precedence'] == 71])}")
    if len(df[df['precedence'] == 71]) > 0:
        print("\nFiles with precedence 71:")
        print(df[df['precedence'] == 71][['system', 'project', 'file', 'operator']])
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo "" | tee -a "$CITATIONS_FILE"
echo "=== COMPLETE ===" | tee -a "$CITATIONS_FILE"
echo "Data saved to:" | tee -a "$CITATIONS_FILE"
echo "- CSV: $SURVEY_FILE" | tee -a "$CITATIONS_FILE"
echo "- Parquet: $OUTPUT_DIR/precedence_survey_${TIMESTAMP}.parquet" | tee -a "$CITATIONS_FILE"
echo "- Citations: $CITATIONS_FILE" | tee -a "$CITATIONS_FILE"
