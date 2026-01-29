{
  description = "Monster Resonance Pipeline - Register capture + Harmonic analysis";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        juliaWithPackages = pkgs.julia.withPackages [
          "FFTW"
          "AbstractFFTs"
          "LinearAlgebra"
          "StaticArrays"
        ];

      in {
        packages = {
          # Register capture tool
          register-capture = pkgs.writeShellScriptBin "capture-registers" ''
            #!/usr/bin/env bash
            set -euo pipefail
            
            PROGRAM="$1"
            OUTPUT="''${2:-registers.json}"
            
            echo "🎯 Capturing registers from: $PROGRAM"
            
            # Run with perf and capture registers
            perf record -e cycles:u --intr-regs=AX,BX,CX,DX,SI,DI,R8,R9,R10,R11,R12,R13,R14,R15 \
              -o perf.data "$PROGRAM"
            
            # Extract register values
            perf script -i perf.data -F ip,sym,iregs | \
              ${pkgs.python3}/bin/python3 ${./scripts/parse_registers.py} > "$OUTPUT"
            
            echo "✅ Registers saved to: $OUTPUT"
            rm -f perf.data
          '';

          # Harmonic analysis tool
          harmonic-analyzer = pkgs.writeShellScriptBin "analyze-harmonics" ''
            #!/usr/bin/env bash
            set -euo pipefail
            
            REGISTERS="$1"
            OUTPUT="''${2:-harmonics.parquet}"
            
            echo "🎵 Analyzing harmonics in: $REGISTERS"
            
            ${juliaWithPackages}/bin/julia ${./scripts/harmonic_analysis.jl} "$REGISTERS" "$OUTPUT"
            
            echo "✅ Harmonics saved to: $OUTPUT"
          '';

          # Monster resonance finder
          monster-resonance = pkgs.writeShellScriptBin "find-monster-resonance" ''
            #!/usr/bin/env bash
            set -euo pipefail
            
            HARMONICS="$1"
            OUTPUT="''${2:-monster_resonance.parquet}"
            
            echo "👹 Finding Monster resonance in: $HARMONICS"
            
            ${pkgs.python3}/bin/python3 ${./scripts/monster_resonance.py} "$HARMONICS" "$OUTPUT"
            
            echo "✅ Monster resonance saved to: $OUTPUT"
          '';

          # Full pipeline
          monster-pipeline = pkgs.writeShellScriptBin "monster-pipeline" ''
            #!/usr/bin/env bash
            set -euo pipefail
            
            PROGRAM="$1"
            BASE="''${2:-pipeline_output}"
            
            echo "╔══════════════════════════════════════════════════════════════╗"
            echo "║  👹 MONSTER RESONANCE PIPELINE 👹                           ║"
            echo "╚══════════════════════════════════════════════════════════════╝"
            echo ""
            
            # Step 1: Capture registers
            echo "📊 Step 1: Capturing registers..."
            ${self.packages.${system}.register-capture}/bin/capture-registers "$PROGRAM" "''${BASE}_registers.json"
            
            # Step 2: Harmonic analysis
            echo ""
            echo "🎵 Step 2: Analyzing harmonics..."
            ${self.packages.${system}.harmonic-analyzer}/bin/analyze-harmonics "''${BASE}_registers.json" "''${BASE}_harmonics.parquet"
            
            # Step 3: Monster resonance
            echo ""
            echo "👹 Step 3: Finding Monster resonance..."
            ${self.packages.${system}.monster-resonance}/bin/find-monster-resonance "''${BASE}_harmonics.parquet" "''${BASE}_monster.parquet"
            
            echo ""
            echo "╔══════════════════════════════════════════════════════════════╗"
            echo "║  ✅ PIPELINE COMPLETE! ✅                                   ║"
            echo "╚══════════════════════════════════════════════════════════════╝"
            echo ""
            echo "📁 Output files:"
            echo "   • ''${BASE}_registers.json"
            echo "   • ''${BASE}_harmonics.parquet"
            echo "   • ''${BASE}_monster.parquet"
            echo ""
          '';
        };

        devShells.default = pkgs.mkShell {
          name = "monster-resonance-pipeline";
          buildInputs = with pkgs; [
            # Performance tools
            linuxPackages.perf
            
            # Julia
            juliaWithPackages
            fftw
            
            # Python
            python3
            python3Packages.pandas
            python3Packages.pyarrow
            python3Packages.numpy
            python3Packages.scipy
            
            # Pipeline tools
            self.packages.${system}.register-capture
            self.packages.${system}.harmonic-analyzer
            self.packages.${system}.monster-resonance
            self.packages.${system}.monster-pipeline
          ];
          
          shellHook = ''
            echo "╔══════════════════════════════════════════════════════════════╗"
            echo "║  👹 MONSTER RESONANCE PIPELINE 👹                           ║"
            echo "╚══════════════════════════════════════════════════════════════╝"
            echo ""
            echo "🎯 Pipeline:"
            echo "   1. Capture registers with perf"
            echo "   2. Apply harmonic analysis (FFT)"
            echo "   3. Find Monster group resonance"
            echo ""
            echo "🚀 Commands:"
            echo "   capture-registers <program> [output.json]"
            echo "   analyze-harmonics <registers.json> [harmonics.parquet]"
            echo "   find-monster-resonance <harmonics.parquet> [monster.parquet]"
            echo "   monster-pipeline <program> [output_base]"
            echo ""
            echo "📊 Example:"
            echo "   monster-pipeline ./my_program test"
            echo ""
            echo "👹 Monster primes:"
            echo "   2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71"
            echo ""
          '';
        };
      }
    );
}
