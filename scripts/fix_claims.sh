#!/bin/bash
# Script to help fix unfounded claims across all documents

echo "🔍 CLAIMS REVIEW - Fixing Unfounded Claims"
echo "=========================================="
echo ""

# Find all markdown files
FILES=$(find . -name "*.md" -not -path "./target/*" -not -path "./.git/*")

echo "📝 Files to review:"
echo "$FILES" | nl
echo ""

# Search for problematic claims
echo "🚨 UNFOUNDED CLAIMS TO FIX:"
echo ""

echo "1. 'Hecke eigenform' claims:"
grep -n "Hecke eigenform\|IS a Hecke\|Hecke eigenvalue" $FILES 2>/dev/null | head -20
echo ""

echo "2. 'Every measurement' claims:"
grep -n "Every measurement\|all measurements" $FILES 2>/dev/null | head -20
echo ""

echo "3. 'Ready to translate ALL' claims:"
grep -n "Ready to translate ALL\|translate ALL LMFDB" $FILES 2>/dev/null | head -20
echo ""

echo "4. 'Everything is equivalent mod 71' claims:"
grep -n "everything is equivalent mod 71\|Everything.*mod 71" $FILES 2>/dev/null | head -20
echo ""

echo "5. 'Overcapacity' claims:"
grep -n "overcapacity\|71\^5" $FILES 2>/dev/null | head -20
echo ""

echo "6. 'Hecke operators preserve' claims:"
grep -n "Hecke operators preserve\|preserve group structure" $FILES 2>/dev/null | head -20
echo ""

echo "7. 'Conway activates' claims:"
grep -n "Conway.*activates\|Conway's name activates" $FILES 2>/dev/null | head -20
echo ""

echo ""
echo "📊 STATISTICS:"
echo "=============="

TOTAL_FILES=$(echo "$FILES" | wc -l)
echo "Total markdown files: $TOTAL_FILES"

HECKE_COUNT=$(grep -l "Hecke eigenform" $FILES 2>/dev/null | wc -l)
echo "Files with 'Hecke eigenform': $HECKE_COUNT"

ALL_LMFDB_COUNT=$(grep -l "ALL LMFDB" $FILES 2>/dev/null | wc -l)
echo "Files with 'ALL LMFDB': $ALL_LMFDB_COUNT"

OVERCAPACITY_COUNT=$(grep -l "overcapacity" $FILES 2>/dev/null | wc -l)
echo "Files with 'overcapacity': $OVERCAPACITY_COUNT"

echo ""
echo "✅ NEXT STEPS:"
echo "=============="
echo "1. Review CLAIMS_REVIEW.md for detailed analysis"
echo "2. Edit each file to fix unfounded claims"
echo "3. Add evidence classification markers (✅ 🔬 📊 💭 ⚠️)"
echo "4. Run this script again to verify fixes"
echo ""
echo "💡 TIP: Use 'git diff' to review changes before committing"
