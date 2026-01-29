{
  description = "Virtual Knuth Pipeline: Literate Web → LLM → Parquet";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          pandas
          pyarrow
          beautifulsoup4
          requests
        ]);
        
        virtualKnuth = pkgs.writeShellScriptBin "virtual-knuth" ''
          export PATH=${pkgs.ollama}/bin:$PATH
          ${pythonEnv}/bin/python3 ${./virtual_knuth.py}
        '';
        
        pipeliteVirtualKnuth = pkgs.writeShellScriptBin "pipelite-virtual-knuth" ''
          export PATH=${pkgs.ollama}/bin:${pythonEnv}/bin:$PATH
          ${./pipelite_virtual_knuth.sh}
        '';
        
      in {
        packages = {
          default = pipeliteVirtualKnuth;
          virtual-knuth = virtualKnuth;
          pipelite-virtual-knuth = pipeliteVirtualKnuth;
        };
        
        apps = {
          default = {
            type = "app";
            program = "${pipeliteVirtualKnuth}/bin/pipelite-virtual-knuth";
          };
          virtual-knuth = {
            type = "app";
            program = "${virtualKnuth}/bin/virtual-knuth";
          };
        };
        
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Lean4
            lean4
            
            # Python + deps
            pythonEnv
            
            # LLM
            ollama
            
            # Documentation
            pandoc
            
            # Utilities
            jq
            ripgrep
            
            # Our tools
            virtualKnuth
            pipeliteVirtualKnuth
          ];
          
          shellHook = ''
            echo "🕸️  Virtual Knuth Pipeline"
            echo "=========================="
            echo ""
            echo "Commands:"
            echo "  pipelite-virtual-knuth  - Full pipeline"
            echo "  virtual-knuth           - LLM review only"
            echo ""
            echo "Pipeline:"
            echo "  Literate Web → Virtual Knuth (LLM) → Parquet"
            echo ""
            echo "Output:"
            echo "  - knuth_reviews.parquet"
            echo "  - language_complexity.parquet"
            echo "  - knuth_metadata.parquet"
            echo "  - huggingface_dataset.json"
            echo ""
            echo "🎯 Ready to feed proofs to virtual Knuth!"
          '';
        };
      }
    );
}
