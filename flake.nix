{
  description = "Monster group order analyzer with LaTeX paper";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # LaTeX paper build (v1)
        monster-paper = pkgs.stdenv.mkDerivation {
          name = "monster-walk-paper";
          src = ./.;
          
          buildInputs = with pkgs; [
            texlive.combined.scheme-full
            pandoc
            python3Packages.pygments
          ];
          
          buildPhase = ''
            pdflatex monster_walk.tex
            bibtex monster_walk || true
            pdflatex monster_walk.tex
            pdflatex monster_walk.tex
            
            pandoc monster_walk.tex \
              -o monster_walk.html \
              --standalone \
              --mathjax \
              --toc \
              --number-sections \
              --css=https://latex.now.sh/style.css \
              --metadata title="The Monster Group Walk and Bott Periodicity"
            
            pandoc monster_walk.tex \
              -o monster_walk_mathml.html \
              --standalone \
              --mathml \
              --toc \
              --number-sections
          '';
          
          installPhase = ''
            mkdir -p $out
            cp monster_walk.pdf $out/ || true
            cp monster_walk.html $out/
            cp monster_walk_mathml.html $out/
            cp monster_walk.tex $out/
          '';
        };
        
        # Literate programming version (v2)
        monster-paper-v2 = pkgs.stdenv.mkDerivation {
          name = "monster-walk-paper-v2-literate";
          src = ./.;
          
          buildInputs = with pkgs; [
            texlive.combined.scheme-full
            pandoc
            python3Packages.pygments
          ];
          
          buildPhase = ''
            pdflatex -shell-escape monster_walk_v2.tex
            pdflatex -shell-escape monster_walk_v2.tex
            pdflatex -shell-escape monster_walk_v2.tex
            
            pandoc monster_walk_v2.tex \
              -o monster_walk_v2.html \
              --standalone \
              --mathjax \
              --toc \
              --number-sections \
              --highlight-style=pygments \
              --css=https://latex.now.sh/style.css \
              --metadata title="The Monster Walk: Literate Programming Edition"
          '';
          
          installPhase = ''
            mkdir -p $out
            cp monster_walk_v2.pdf $out/
            cp monster_walk_v2.html $out/
            cp monster_walk_v2.tex $out/
          '';
        };
        
        # Interactive Pyodide site
        monster-pyodide-site = pkgs.stdenv.mkDerivation {
          name = "monster-pyodide-site";
          src = ./.;
          
          buildPhase = ''
            mkdir -p site
            cp web/pyodide.html site/index.html
            cp web/style.css site/
            cp web/index.html site/wasm-demo.html
            
            # Copy WASM if built
            if [ -d wasm/pkg ]; then
              cp -r wasm/pkg site/wasm/
            fi
          '';
          
          installPhase = ''
            mkdir -p $out
            cp -r site/* $out/
          '';
        };
        
        # AI sampler with models
        monster-ai-sampler = pkgs.rustPlatform.buildRustPackage {
          pname = "monster-ai-sampler";
          version = "0.1.0";
          src = ./ai-sampler;
          
          cargoLock = {
            lockFile = ./ai-sampler/Cargo.lock;
          };
          
          nativeBuildInputs = with pkgs; [
            pkg-config
          ];
          
          buildInputs = with pkgs; [
            openssl
            chromium
          ];
          
          postInstall = ''
            mkdir -p $out/share/monster-ai-sampler
            echo "AI Sampler installed"
          '';
        };
        
        # Complete AI environment
        monster-ai-env = pkgs.buildEnv {
          name = "monster-ai-env";
          paths = with pkgs; [
            ollama
            monster-ai-sampler
            chromium
            python311
            python311Packages.transformers
            python311Packages.torch
            python311Packages.pillow
          ];
        };
        
      in
      {
        packages = {
          paper = monster-paper;
          paper-v2 = monster-paper-v2;
          pyodide-site = monster-pyodide-site;
          ai-sampler = monster-ai-sampler;
          ai-env = monster-ai-env;
          default = monster-pyodide-site;
        };
        
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            cargo
            rustc
            rustfmt
            clippy
            texlive.combined.scheme-full
            pandoc
            python3Packages.pygments
            chromium
            act
            pkg-config
            openssl
          ];
          
          shellHook = ''
            export CHROME_BIN=${pkgs.chromium}/bin/chromium
            export PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=1
            export MISTRALRS_CACHE=$HOME/.cache/mistral.rs
            
            echo "Monster Walk Development Environment"
            echo "===================================="
            echo "Pure Rust AI with mistral.rs"
            echo ""
            echo "Available commands:"
            echo "  cargo run --bin sample-with-mistralrs"
            echo "  nix build .#paper-v2"
            echo "  nix build .#pyodide-site"
            echo ""
            echo "Models cached in: $MISTRALRS_CACHE"
          '';
        };
      }
    );
}
