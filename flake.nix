{
  description = "Monster Group LMFDB with Literate Programming";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    zkprologml.url = "github:meta-introspector/zkprologml";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, zkprologml }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        
        rustToolchain = pkgs.rust-bin.stable.latest.default;
        
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          numpy
          scipy
          sympy
          pandas
          matplotlib
          jupyter
          huggingface-hub
          pillow
          requests
        ]);
        
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Rust
            rustToolchain
            pkgs.cargo
            pkgs.rustc
            
            # Python
            pythonEnv
            
            # Database
            pkgs.postgresql
            
            # Performance
            pkgs.perf
            
            # Tools
            pkgs.git
            pkgs.jq
            pkgs.act
            
            # Full LaTeX environment
            pkgs.texlive.combined.scheme-full
            
            # Literate programming
            pkgs.noweb          # Universal literate programming
            # cweb included in texlive
            
            # Document tools
            pkgs.pandoc
            pkgs.poppler-utils  # pdftoppm, pdftotext
            pkgs.imagemagick
            pkgs.ghostscript
            pkgs.qpdf
            
            # Diagrams
            pkgs.graphviz
            pkgs.plantuml
            
            # Dataset tools from zkprologml
            zkprologml.packages.${system}.default or pkgs.hello
          ];
          
          shellHook = ''
            echo "🔬 Monster Group Walk Analysis Environment"
            echo "=========================================="
            echo ""
            echo "📄 Paper Review:"
            echo "  cargo run --release --bin review_paper PAPER.tex"
            echo "  ./review_paper.sh"
            echo ""
            echo "📊 Core Analysis:"
            echo "  cargo run --release              # Monster Walk"
            echo "  cd MonsterLean && lake build     # Lean4 proofs"
            echo ""
            echo "📝 LaTeX:"
            echo "  pdflatex PAPER.tex"
            echo "  pdftoppm -png PAPER.pdf page"
            echo ""
            echo "🔍 Review System:"
            echo "  Requires: ollama with llava model"
            echo "  Install: curl -fsSL https://ollama.com/install.sh | sh"
            echo "  Pull model: ollama pull llava"
            echo ""
          '';
        };
      }
    );
}
