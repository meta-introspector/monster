#!/usr/bin/env bash

echo "🔬 ANALYZING METACOQ STRUCTURE"
echo "=============================="
echo ""

# Check if we can access MetaCoq
if [ -d "$HOME/.opam" ]; then
    echo "📦 Checking for MetaCoq installation..."
    opam list | grep -i metacoq || echo "MetaCoq not found in opam"
fi

# Alternative: Check Coq libraries
if [ -d "$HOME/.coq" ]; then
    echo "📚 Checking Coq libraries..."
    find ~/.coq -name "*metacoq*" -o -name "*unimath*" 2>/dev/null | head -5
fi

# Check for local installations
echo ""
echo "🔍 Searching for MetaCoq/UniMath sources..."
find ~ -maxdepth 3 -type d -name "*metacoq*" -o -name "*unimath*" -o -name "*UniMath*" 2>/dev/null | head -10

echo ""
echo "💡 We need to:"
echo "  1. Clone MetaCoq repository"
echo "  2. Clone UniMath repository"
echo "  3. Analyze their AST structure"
echo "  4. Measure tree depth (looking for 46 levels!)"
echo "  5. Count Monster primes in their code"
