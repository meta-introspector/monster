#!/bin/bash
# Convert PAPER.md to PDF and PNG using pandoc

echo "📄 CONVERTING PAPER TO PDF AND PNG"
echo "===================================="
echo

# Check for pandoc
if ! command -v pandoc &> /dev/null; then
    echo "❌ pandoc not found"
    echo "Install with: sudo apt-get install pandoc texlive-latex-base texlive-latex-extra"
    exit 1
fi

# Convert to PDF
echo "Step 1: Converting Markdown to PDF..."
pandoc PAPER.md \
    -o PAPER.pdf \
    --pdf-engine=pdflatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    --toc \
    --number-sections \
    2>&1 | grep -v "warning"

if [ -f PAPER.pdf ]; then
    echo "✅ Created: PAPER.pdf"
    SIZE=$(du -h PAPER.pdf | cut -f1)
    echo "   Size: $SIZE"
else
    echo "❌ Failed to create PDF"
    exit 1
fi

echo

# Check for pdftoppm
if ! command -v pdftoppm &> /dev/null; then
    echo "❌ pdftoppm not found"
    echo "Install with: sudo apt-get install poppler-utils"
    exit 1
fi

# Convert PDF to PNG
echo "Step 2: Converting PDF to PNG images..."
pdftoppm -png -r 150 PAPER.pdf PAPER_page

if ls PAPER_page-*.png 1> /dev/null 2>&1; then
    NUM=$(ls PAPER_page-*.png | wc -l)
    echo "✅ Created: $NUM PNG images"
    ls PAPER_page-*.png | head -5
    if [ $NUM -gt 5 ]; then
        echo "   ... and $((NUM - 5)) more"
    fi
else
    echo "❌ Failed to create PNG images"
    exit 1
fi

echo
echo "===================================="
echo "✅ CONVERSION COMPLETE"
echo "===================================="
echo
echo "Files created:"
echo "  - PAPER.pdf (full document)"
echo "  - PAPER_page-*.png (page images)"
echo
echo "Next: Run vision model review"
echo "  ./review_paper_with_vision.sh"
