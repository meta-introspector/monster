#!/bin/bash
# Review paper with vision model - auto-setup

echo "👁️  PAPER VISION REVIEW"
echo "======================"
echo

# Step 1: Check for llava
if ! ollama list | grep -q "llava"; then
    echo "📥 Pulling llava vision model..."
    ollama pull llava
    echo "✅ llava ready"
fi

echo

# Step 2: Create images if needed
if ! ls PAPER_page-*.png 1> /dev/null 2>&1; then
    echo "📄 No images found. Let me check what we have..."
    
    # Check for PAPER.md
    if [ -f PAPER.md ]; then
        echo "Found PAPER.md - converting to images..."
        
        # Use pandoc in nix environment
        nix develop --command bash -c "
            pandoc PAPER.md -o PAPER.pdf --pdf-engine=xelatex 2>&1 | grep -v warning
            if [ -f PAPER.pdf ]; then
                pdftoppm -png -r 150 PAPER.pdf PAPER_page
                echo '✅ Created PNG images'
            fi
        "
    else
        echo "❌ No PAPER.md found"
        exit 1
    fi
fi

# Count images
NUM=$(ls PAPER_page-*.png 2>/dev/null | wc -l)
echo "📊 Found $NUM pages to review"
echo

# Step 3: Review with llava
mkdir -p vision_reviews

echo "🔍 Reviewing with llava..."
echo

for img in PAPER_page-*.png; do
    PAGE=$(basename "$img" .png | sed 's/PAPER_page-//')
    OUTPUT="vision_reviews/page_${PAGE}.txt"
    
    echo "Page $PAGE..."
    
    ollama run llava "$img" "You are reviewing a research paper on the Monster Group Neural Network.

For this page, analyze:
1. **Mathematical Correctness**: Are formulas correct?
2. **Clarity**: Is the presentation clear?
3. **Completeness**: What's missing?
4. **Diagrams**: Are visualizations needed?
5. **Errors**: Any mistakes or inconsistencies?

Be critical and specific." > "$OUTPUT" 2>&1
    
    echo "  ✅ $OUTPUT"
done

echo
echo "======================"
echo "✅ REVIEW COMPLETE"
echo "======================"
echo
echo "📁 Results: vision_reviews/"
echo
echo "View first review:"
echo "  cat vision_reviews/page_1.txt"
echo
echo "View all:"
echo "  cat vision_reviews/*.txt"
