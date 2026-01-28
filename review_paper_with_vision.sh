#!/bin/bash
# Review paper with LLaVA vision model

echo "🔍 REVIEWING PAPER WITH LLAVA VISION MODEL"
echo "=========================================="
echo

# Check for PNG images
if ! ls PAPER_page-*.png 1> /dev/null 2>&1; then
    echo "❌ No PNG images found"
    echo "Run: python3 convert_paper_to_visual.py first"
    exit 1
fi

# Count images
NUM_IMAGES=$(ls PAPER_page-*.png | wc -l)
echo "Found $NUM_IMAGES pages to review"
echo

# Create output directory
mkdir -p vision_reviews

# Review each page
for img in PAPER_page-*.png; do
    PAGE=$(basename "$img" .png | sed 's/PAPER_page-//')
    echo "Reviewing page $PAGE..."
    
    # Call mistral.rs with LLaVA
    # Adjust this command based on your mistral.rs setup
    mistralrs-server \
        --model llava \
        --image "$img" \
        --prompt "Review this page of a research paper. Check for:
1. Mathematical correctness
2. Clarity of presentation
3. Missing diagrams or visualizations
4. Inconsistencies or errors
5. Suggestions for improvement

Be critical and thorough." \
        > "vision_reviews/page_${PAGE}_review.txt" 2>&1
    
    echo "✅ Saved: vision_reviews/page_${PAGE}_review.txt"
done

echo
echo "=========================================="
echo "✅ REVIEW COMPLETE"
echo "=========================================="
echo
echo "Results in: vision_reviews/"
echo
echo "To summarize:"
echo "  cat vision_reviews/*.txt > vision_reviews/FULL_REVIEW.txt"
