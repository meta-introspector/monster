# Makefile for Monster Group Literate Proof

.PHONY: all clean code docs pdf png test

all: code docs pdf png

# Extract executable code from literate source
code: monster_proof.rs
	@echo "✅ Code extracted"

monster_proof.rs: monster_proof.nw
	notangle -R'*' monster_proof.nw > monster_proof.rs

# Extract LaTeX documentation
docs: monster_proof.tex
	@echo "✅ Documentation extracted"

monster_proof.tex: monster_proof.nw
	noweave -latex -delay monster_proof.nw > monster_proof.tex

# Compile PDF
pdf: monster_proof.pdf
	@echo "✅ PDF compiled"

monster_proof.pdf: monster_proof.tex
	pdflatex monster_proof.tex
	pdflatex monster_proof.tex  # Run twice for references

# Convert PDF to PNG images
png: monster_proof_page-1.png
	@echo "✅ PNG images created"

monster_proof_page-1.png: monster_proof.pdf
	pdftoppm -png -r 150 monster_proof.pdf monster_proof_page

# Test extracted code
test: monster_proof.rs
	rustc --test monster_proof.rs -o monster_proof_test
	./monster_proof_test
	@echo "✅ All tests passed"

# Clean generated files
clean:
	rm -f monster_proof.rs
	rm -f monster_proof.tex
	rm -f monster_proof.pdf
	rm -f monster_proof.aux monster_proof.log monster_proof.toc
	rm -f monster_proof_page-*.png
	rm -f monster_proof_test
	@echo "✅ Cleaned"

# Full workflow
workflow: all test
	@echo ""
	@echo "🎉 COMPLETE WORKFLOW FINISHED"
	@echo "=============================="
	@echo ""
	@echo "Generated files:"
	@echo "  - monster_proof.rs (executable code)"
	@echo "  - monster_proof.tex (LaTeX documentation)"
	@echo "  - monster_proof.pdf (compiled paper)"
	@echo "  - monster_proof_page-*.png (page images)"
	@echo ""
	@echo "All tests passed ✅"
	@echo ""

# Help
help:
	@echo "Monster Group Literate Proof - Makefile"
	@echo "========================================"
	@echo ""
	@echo "Targets:"
	@echo "  make all       - Extract code and docs, compile PDF, create PNGs"
	@echo "  make code      - Extract executable Rust code"
	@echo "  make docs      - Extract LaTeX documentation"
	@echo "  make pdf       - Compile PDF"
	@echo "  make png       - Convert PDF to PNG images"
	@echo "  make test      - Run tests on extracted code"
	@echo "  make workflow  - Complete workflow with tests"
	@echo "  make clean     - Remove generated files"
	@echo "  make help      - Show this help"
	@echo ""
	@echo "Literate programming with noweb:"
	@echo "  Edit: monster_proof.nw"
	@echo "  Build: make workflow"
	@echo ""
