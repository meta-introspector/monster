#!/usr/bin/env python3
"""
Convert PAPER.md to LaTeX, PDF, and PNG for vision model review
"""

import subprocess
import os
from pathlib import Path

print("📄 CONVERTING PAPER TO VISUAL FORMATS")
print("=" * 60)
print()

# Step 1: Convert Markdown to LaTeX
print("Step 1: Converting Markdown to LaTeX...")

latex_header = r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}

\newtheorem{theorem}{Theorem}
\newtheorem{definition}{Definition}
\newtheorem{proof}{Proof}

\title{Monster Group Neural Network: A Literate Proof}
\author{Meta-Introspector Research}
\date{January 28, 2026}

\begin{document}
\maketitle
\tableofcontents
\newpage

"""

latex_footer = r"""
\end{document}
"""

# Read paper
with open('PAPER.md') as f:
    paper = f.read()

# Simple Markdown to LaTeX conversion
latex_body = paper

# Convert headers
latex_body = latex_body.replace('## ', r'\section{')
latex_body = latex_body.replace('### ', r'\subsection{')
latex_body = latex_body.replace('#### ', r'\subsubsection{')

# Close section headers
import re
latex_body = re.sub(r'\\(sub)*section\{([^}]+)\}', r'\\\1section{\2}', latex_body)

# Convert code blocks
latex_body = re.sub(r'```(\w+)\n(.*?)```', 
                    r'\\begin{lstlisting}[language=\1]\n\2\\end{lstlisting}',
                    latex_body, flags=re.DOTALL)

latex_body = re.sub(r'```\n(.*?)```', 
                    r'\\begin{verbatim}\n\1\\end{verbatim}',
                    latex_body, flags=re.DOTALL)

# Convert inline code
latex_body = re.sub(r'`([^`]+)`', r'\\texttt{\1}', latex_body)

# Convert bold
latex_body = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', latex_body)

# Convert italic
latex_body = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', latex_body)

# Escape special characters
latex_body = latex_body.replace('_', r'\_')
latex_body = latex_body.replace('%', r'\%')
latex_body = latex_body.replace('&', r'\&')
latex_body = latex_body.replace('#', r'\#')

# Write LaTeX file
latex_content = latex_header + latex_body + latex_footer

with open('PAPER.tex', 'w') as f:
    f.write(latex_content)

print("✅ Created: PAPER.tex")
print()

# Step 2: Convert LaTeX to PDF
print("Step 2: Converting LaTeX to PDF...")

try:
    result = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', 'PAPER.tex'],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode == 0:
        print("✅ Created: PAPER.pdf")
    else:
        print("⚠️  pdflatex had warnings, but PDF may be created")
        print(f"   Check PAPER.log for details")
except FileNotFoundError:
    print("❌ pdflatex not found. Install with: sudo apt-get install texlive-latex-base")
except subprocess.TimeoutExpired:
    print("❌ pdflatex timed out")
except Exception as e:
    print(f"❌ Error: {e}")

print()

# Step 3: Convert PDF to PNG images
print("Step 3: Converting PDF to PNG images...")

if Path('PAPER.pdf').exists():
    try:
        # Convert each page to PNG
        result = subprocess.run(
            ['pdftoppm', '-png', '-r', '150', 'PAPER.pdf', 'PAPER_page'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            # Count generated images
            images = list(Path('.').glob('PAPER_page-*.png'))
            print(f"✅ Created: {len(images)} PNG images")
            for img in sorted(images)[:5]:
                print(f"   - {img}")
            if len(images) > 5:
                print(f"   ... and {len(images) - 5} more")
        else:
            print("❌ pdftoppm failed")
    except FileNotFoundError:
        print("❌ pdftoppm not found. Install with: sudo apt-get install poppler-utils")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ PAPER.pdf not found, skipping PNG conversion")

print()

# Step 4: Create vision model review script
print("Step 4: Creating vision model review script...")

review_script = """#!/bin/bash
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
    mistralrs-server \\
        --model llava \\
        --image "$img" \\
        --prompt "Review this page of a research paper. Check for:
1. Mathematical correctness
2. Clarity of presentation
3. Missing diagrams or visualizations
4. Inconsistencies or errors
5. Suggestions for improvement

Be critical and thorough." \\
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
"""

with open('review_paper_with_vision.sh', 'w') as f:
    f.write(review_script)

os.chmod('review_paper_with_vision.sh', 0o755)

print("✅ Created: review_paper_with_vision.sh")
print()

# Step 5: Summary
print("=" * 60)
print("CONVERSION SUMMARY")
print("=" * 60)
print()

files_created = []
if Path('PAPER.tex').exists():
    files_created.append('PAPER.tex (LaTeX source)')
if Path('PAPER.pdf').exists():
    files_created.append('PAPER.pdf (PDF document)')
    
images = list(Path('.').glob('PAPER_page-*.png'))
if images:
    files_created.append(f'{len(images)} PNG images (PAPER_page-*.png)')

files_created.append('review_paper_with_vision.sh (review script)')

print("Files created:")
for f in files_created:
    print(f"  ✅ {f}")

print()
print("Next steps:")
print("  1. Check PAPER.pdf for rendering issues")
print("  2. View PNG images: eog PAPER_page-*.png")
print("  3. Run vision review: ./review_paper_with_vision.sh")
print()

# Create a simple HTML viewer
print("Creating HTML viewer...")

html_viewer = """<!DOCTYPE html>
<html>
<head>
    <title>Monster Paper - Visual Review</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f0f0f0; }
        .page { margin: 20px auto; max-width: 800px; background: white; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        img { width: 100%; border: 1px solid #ddd; }
        h1 { color: #333; }
        .nav { text-align: center; margin: 20px; }
        .nav button { padding: 10px 20px; margin: 5px; font-size: 16px; }
    </style>
</head>
<body>
    <h1>Monster Group Neural Network - Visual Review</h1>
    <div class="nav">
        <button onclick="prevPage()">← Previous</button>
        <span id="pageNum">Page 1</span>
        <button onclick="nextPage()">Next →</button>
    </div>
    <div class="page">
        <img id="pageImg" src="PAPER_page-1.png" alt="Page 1">
    </div>
    <script>
        let currentPage = 1;
        const totalPages = """ + str(len(images)) + """;
        
        function updatePage() {
            document.getElementById('pageImg').src = `PAPER_page-${currentPage}.png`;
            document.getElementById('pageNum').textContent = `Page ${currentPage} of ${totalPages}`;
        }
        
        function nextPage() {
            if (currentPage < totalPages) {
                currentPage++;
                updatePage();
            }
        }
        
        function prevPage() {
            if (currentPage > 1) {
                currentPage--;
                updatePage();
            }
        }
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight') nextPage();
            if (e.key === 'ArrowLeft') prevPage();
        });
    </script>
</body>
</html>
"""

with open('PAPER_viewer.html', 'w') as f:
    f.write(html_viewer)

print("✅ Created: PAPER_viewer.html")
print("   Open in browser to view pages")
print()
