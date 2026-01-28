#!/bin/bash
# Review paper with LLaVA vision model

echo "🔍 REVIEWING PAPER WITH LLAVA"
echo "=============================="
echo

# Check if mistral.rs is available
if ! command -v mistralrs &> /dev/null; then
    echo "❌ mistralrs not found"
    echo "Please install mistral.rs first"
    exit 1
fi

# Create review prompt
cat > review_prompt.txt << 'EOF'
You are a critical reviewer of a research paper on neural networks and group theory.

Please review the following paper and provide detailed feedback on:

1. CLARITY (1-10): How clear is the presentation?
2. ACCURACY (1-10): Are the claims accurate?
3. COMPLETENESS (1-10): What's missing?
4. ISSUES: What problems do you see?
5. DIAGRAMS: What visualizations are needed?

Be critical and thorough. Point out any errors, inconsistencies, or unclear sections.

Paper sections to review:
- Architecture (71-layer autoencoder)
- J-Invariant World (unified object model)
- Compression proofs (23× ratio)
- Equivalence proofs (Python ≡ Rust)
- 16 theorems

Known issues we found:
1. Shard count: claimed 70, found 71
2. Parameter count: 9,690 vs 9,452
3. Architecture string not found
4. J-invariant formula not verified

Please verify these and find any additional issues.
EOF

echo "📄 Review prompt created"
echo

# Review PAPER.md
echo "Reviewing PAPER.md..."
echo

# Run mistral.rs with LLaVA
# Adjust command based on your mistral.rs setup
mistralrs-server \
    --model llava \
    --prompt "$(cat review_prompt.txt)" \
    --file PAPER.md \
    > llava_review_paper.txt 2>&1

echo "✅ Paper review saved to: llava_review_paper.txt"
echo

# Review VISUAL_SUMMARY.md
echo "Reviewing VISUAL_SUMMARY.md..."
echo

mistralrs-server \
    --model llava \
    --prompt "Review these ASCII diagrams. Are they clear? What improvements are needed?" \
    --file VISUAL_SUMMARY.md \
    > llava_review_visual.txt 2>&1

echo "✅ Visual review saved to: llava_review_visual.txt"
echo

# Review CRITICAL_EVALUATION.md
echo "Reviewing CRITICAL_EVALUATION.md..."
echo

mistralrs-server \
    --model llava \
    --prompt "Review this self-evaluation. Did we miss any issues? Are our assessments correct?" \
    --file CRITICAL_EVALUATION.md \
    > llava_review_evaluation.txt 2>&1

echo "✅ Evaluation review saved to: llava_review_evaluation.txt"
echo

echo "=============================="
echo "✅ REVIEW COMPLETE"
echo "=============================="
echo
echo "Results:"
echo "  - llava_review_paper.txt"
echo "  - llava_review_visual.txt"
echo "  - llava_review_evaluation.txt"
echo
echo "Next: Analyze results and fix issues"
