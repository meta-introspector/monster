#!/usr/bin/env bash
# Knuth-style TANGLE: Extract code from literate web

set -e

echo "🕸️  TANGLE: Extracting code from literate web..."
echo ""

# Extract Lean4 code
echo "📝 Extracting Lean4 code..."
grep -A 1000 '<div class="code-block lean">' literate_web.html | \
  grep -B 1000 '</div>' | \
  sed 's/<[^>]*>//g' | \
  sed '/^$/d' | \
  grep -v "Copy" > extracted_proof.lean

echo "✓ Extracted to: extracted_proof.lean"
echo ""

# Verify it matches original
echo "🔍 Verifying against original..."
if diff -q MonsterLean/CrossLanguageComplexity.lean extracted_proof.lean > /dev/null 2>&1; then
    echo "✓ Perfect match! Literate web is faithful to source."
else
    echo "⚠️  Differences found (expected - literate web shows excerpts)"
    echo "   Original: MonsterLean/CrossLanguageComplexity.lean"
    echo "   Extracted: extracted_proof.lean"
fi
echo ""

# Generate PDF (if pandoc available)
if command -v pandoc &> /dev/null; then
    echo "📄 Generating PDF..."
    pandoc literate_web.html -o literate_proof.pdf \
        --pdf-engine=xelatex \
        --metadata title="Cross-Language Complexity via Monster Layers" \
        --metadata author="Meta-Introspector Project" \
        --metadata date="$(date +%Y-%m-%d)" \
        2>/dev/null && echo "✓ PDF: literate_proof.pdf" || echo "⚠️  PDF generation failed"
else
    echo "ℹ️  Install pandoc for PDF generation"
fi
echo ""

# Open in browser
echo "🌐 Opening in browser..."
if command -v xdg-open &> /dev/null; then
    xdg-open literate_web.html &
elif command -v open &> /dev/null; then
    open literate_web.html &
else
    echo "   Open manually: file://$(pwd)/literate_web.html"
fi

echo ""
echo "✅ TANGLE complete!"
echo ""
echo "📚 Knuth Literate Programming:"
echo "   WEB (literate source): literate_web.html"
echo "   TANGLE (code extract): extracted_proof.lean"
echo "   WEAVE (documentation): literate_web.html (self-documenting)"
