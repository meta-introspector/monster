{
  description = "Monster Group LMFDB with Literate Programming";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
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
            pkgs.poppler_utils  # pdftoppm
            pkgs.imagemagick
            pkgs.ghostscript
            pkgs.qpdf
            
            # Diagrams
            pkgs.graphviz
            pkgs.plantuml
          ];
          
          shellHook = ''
            echo "🔬 Monster Group LMFDB Analysis Environment"
            echo "============================================"
            echo ""
            echo "📊 Analysis:"
            echo "  cargo run --bin abelian_variety"
            echo "  python3 analyze_lmfdb_source.py"
            echo ""
            echo "📝 LaTeX & Literate Programming:"
            echo "  pdflatex, xelatex, lualatex"
            echo "  noweb - Literate programming"
            echo "  cweb - C literate programming"
            echo "  pandoc - Document conversion"
            echo ""
            echo "🔄 Workflow:"
            echo "  notangle file.nw > file.rs    # Extract code"
            echo "  noweave -latex file.nw > file.tex  # Extract docs"
            echo "  pdflatex file.tex             # Compile PDF"
            echo "  pdftoppm -png file.pdf page   # Convert to PNG"
            echo ""
          '';
        };
      }
    );
}
