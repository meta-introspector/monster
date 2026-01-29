{
  description = "Knuth Literate Web Pipeline for Monster Group Proofs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Knuth literate web builder
        knuthWeb = pkgs.writeShellScriptBin "knuth-web" ''
          ${./knuth_pipeline.sh}
        '';
        
        # TANGLE - extract code
        tangle = pkgs.writeShellScriptBin "tangle" ''
          ${./tangle_literate.sh}
        '';
        
        # WEAVE - generate docs (already HTML)
        weave = pkgs.writeShellScriptBin "weave" ''
          echo "📖 WEAVE: literate_web.html is self-documenting!"
          echo "✓ No additional weaving needed"
        '';
        
      in {
        packages = {
          default = knuthWeb;
          knuth-web = knuthWeb;
          tangle = tangle;
          weave = weave;
        };
        
        apps = {
          default = {
            type = "app";
            program = "${knuthWeb}/bin/knuth-web";
          };
          tangle = {
            type = "app";
            program = "${tangle}/bin/tangle";
          };
          weave = {
            type = "app";
            program = "${weave}/bin/weave";
          };
        };
        
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Lean4 toolchain
            lean4
            
            # Documentation tools
            pandoc
            texlive.combined.scheme-full
            
            # Web tools
            python3
            nodejs
            
            # Utilities
            jq
            ripgrep
            
            # Our tools
            knuthWeb
            tangle
            weave
          ];
          
          shellHook = ''
            echo "🕸️  Knuth Literate Web Environment"
            echo "=================================="
            echo ""
            echo "Commands:"
            echo "  knuth-web  - Run complete pipeline"
            echo "  tangle     - Extract code (TANGLE)"
            echo "  weave      - Generate docs (WEAVE)"
            echo ""
            echo "Files:"
            echo "  index.html          - Landing page"
            echo "  interactive_viz.html - Visualization"
            echo "  literate_web.html   - Complete proof"
            echo ""
            echo "🎯 Ready to prove Coq ≃ Lean4 ≃ Rust!"
          '';
        };
      }
    );
}
