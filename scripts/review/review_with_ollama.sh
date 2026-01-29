#!/bin/bash
# Review PAPER.md with vision model using existing tools

echo "👁️  REVIEWING PAPER WITH VISION MODEL"
echo "====================================="
echo

# Check if we have PNG images
if ! ls PAPER_page-*.png 1> /dev/null 2>&1; then
    echo "No PNG images found. Creating them..."
    
    # Try to convert PAPER.md to images
    if command -v pandoc &> /dev/null; then
        echo "Converting PAPER.md to PDF..."
        pandoc PAPER.md -o PAPER.pdf --pdf-engine=xelatex 2>&1 | grep -v "warning"
        
        if [ -f PAPER.pdf ]; then
            echo "✅ PDF created"
            
            if command -v pdftoppm &> /dev/null; then
                echo "Converting PDF to PNG..."
                pdftoppm -png -r 150 PAPER.pdf PAPER_page
                echo "✅ PNG images created"
            fi
        fi
    fi
fi

# Count images
NUM_IMAGES=$(ls PAPER_page-*.png 2>/dev/null | wc -l)

if [ $NUM_IMAGES -eq 0 ]; then
    echo "❌ No images available for review"
    echo "Please run: nix develop --command make png"
    exit 1
fi

echo "Found $NUM_IMAGES pages to review"
echo

# Create output directory
mkdir -p vision_reviews

# Review with ollama (if available)
if command -v ollama &> /dev/null; then
    echo "Using ollama for vision analysis..."
    
    for img in PAPER_page-*.png; do
        PAGE=$(basename "$img" .png | sed 's/PAPER_page-//')
        echo "Reviewing page $PAGE..."
        
        ollama run llava "$img" "Review this page of a research paper. Check for:
1. Mathematical correctness
2. Clarity of presentation  
3. Missing diagrams
4. Inconsistencies
5. Suggestions for improvement" > "vision_reviews/page_${PAGE}_review.txt" 2>&1
        
        echo "✅ Saved: vision_reviews/page_${PAGE}_review.txt"
    done
else
    echo "⚠️  ollama not found"
    echo "Install: curl https://ollama.ai/install.sh | sh"
    echo "Then: ollama pull llava"
fi

echo
echo "====================================="
echo "✅ REVIEW COMPLETE"
echo "====================================="
echo
echo "Results in: vision_reviews/"
echo
echo "View summary:"
echo "  cat vision_reviews/*.txt | head -100"
