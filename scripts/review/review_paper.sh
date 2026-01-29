#!/bin/bash
# Reproducible paper review workflow

set -e  # Exit on error

echo "📄 Monster Walk Paper Review"
echo "=============================="
echo ""

# Step 1: Compile LaTeX to PDF
echo "Step 1: Compiling PAPER.tex to PDF..."
cd /home/mdupont/experiments/monster
pdflatex -interaction=nonstopmode PAPER.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode PAPER.tex > /dev/null 2>&1  # Run twice for references
echo "✓ PDF generated: PAPER.pdf"
echo ""

# Step 2: Convert PDF to images for review
echo "Step 2: Converting PDF to images..."
mkdir -p paper_images
pdftoppm PAPER.pdf paper_images/page -png
echo "✓ Images generated in paper_images/"
echo ""

# Step 3: Run multi-level review
echo "Step 3: Running multi-level review (21 perspectives)..."
python3 multi_level_review.py PAPER.pdf
echo "✓ Reviews generated in multi_level_reviews/"
echo ""

# Step 4: Summary
echo "=============================="
echo "✅ Review Complete!"
echo ""
echo "📖 View results:"
echo "  - Index: cat multi_level_reviews/INDEX.md"
echo "  - Page 1: cat multi_level_reviews/page_01_synthesis.md"
echo "  - Page 2: cat multi_level_reviews/page_02_synthesis.md"
echo "  - Page 3: cat multi_level_reviews/page_03_synthesis.md"
echo ""
echo "🔍 Check for issues:"
echo "  grep -i 'error\\|warning\\|issue' multi_level_reviews/*.md"
