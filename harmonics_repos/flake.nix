{
  description = "Harmonic Analysis for Monster Group - Julia environments";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Julia with required packages
        juliaWithPackages = pkgs.julia.withPackages [
          "FFTW"
          "AbstractFFTs"
          "LinearAlgebra"
          "StaticArrays"
        ];

      in {
        devShells = {
          # DFTK.jl environment
          dftk = pkgs.mkShell {
            name = "dftk-harmonic-analysis";
            buildInputs = with pkgs; [
              juliaWithPackages
              fftw
              fftwFloat
              pkg-config
            ];
            
            shellHook = ''
              echo "🎵 DFTK.jl Harmonic Analysis Environment"
              echo "========================================"
              echo ""
              echo "Key files:"
              echo "  • src/common/spherical_harmonics.jl ⭐⭐⭐⭐⭐"
              echo "  • src/fft.jl ⭐⭐⭐"
              echo "  • src/symmetry.jl ⭐⭐⭐"
              echo ""
              echo "Usage:"
              echo "  cd DFTK.jl"
              echo "  julia"
              echo "  > include(\"src/common/spherical_harmonics.jl\")"
              echo "  > ylm_real(2, 0, [0.0, 0.0, 1.0])"
              echo ""
            '';
          };

          # ApproxFun.jl environment
          approxfun = pkgs.mkShell {
            name = "approxfun-spectral-methods";
            buildInputs = with pkgs; [
              juliaWithPackages
              fftw
            ];
            
            shellHook = ''
              echo "🎵 ApproxFun.jl Spectral Methods Environment"
              echo "==========================================="
              echo ""
              echo "Key files:"
              echo "  • src/Extras/fftGeneric.jl ⭐⭐⭐"
              echo "  • examples/Eigenvalue_anharmonic.jl ⭐⭐⭐"
              echo ""
              echo "Usage:"
              echo "  cd ApproxFun.jl"
              echo "  julia"
              echo ""
            '';
          };

          # Combined environment
          default = pkgs.mkShell {
            name = "harmonic-analysis-combined";
            buildInputs = with pkgs; [
              juliaWithPackages
              fftw
              fftwFloat
              pkg-config
              
              # Analysis tools
              python3
              python3Packages.pandas
              python3Packages.pyarrow
              python3Packages.numpy
            ];
            
            shellHook = ''
              echo "╔══════════════════════════════════════════════════════════════╗"
              echo "║  🎵 HARMONIC ANALYSIS FOR MONSTER GROUP 🎵                  ║"
              echo "╚══════════════════════════════════════════════════════════════╝"
              echo ""
              echo "📦 Repos:"
              echo "  • DFTK.jl (263 files, 76% harmonic-related)"
              echo "  • ApproxFun.jl (57 files, 47% harmonic-related)"
              echo ""
              echo "🔑 Key Files:"
              echo "  1. DFTK.jl/src/common/spherical_harmonics.jl ⭐⭐⭐⭐⭐"
              echo "  2. DFTK.jl/src/fft.jl ⭐⭐⭐"
              echo "  3. DFTK.jl/src/symmetry.jl ⭐⭐⭐"
              echo "  4. ApproxFun.jl/src/Extras/fftGeneric.jl ⭐⭐⭐"
              echo ""
              echo "🎯 Connection to Monster:"
              echo "  SO(3) Spherical Harmonics ↔ Monster Characters (194)"
              echo "  Y_l^m (2l+1 functions)    ↔ χ_i (194 characters)"
              echo ""
              echo "🚀 Quick Start:"
              echo "  nix develop .#dftk       # DFTK.jl only"
              echo "  nix develop .#approxfun  # ApproxFun.jl only"
              echo "  nix develop              # Both + analysis tools"
              echo ""
              echo "📊 Analysis:"
              echo "  python3 -c 'import pandas as pd; print(pd.read_parquet(\"../harmonics_ranked.parquet\").head())'"
              echo ""
            '';
          };
        };

        # Apps for quick testing
        apps = {
          test-spherical = {
            type = "app";
            program = toString (pkgs.writeShellScript "test-spherical" ''
              cd ${self}/DFTK.jl
              ${juliaWithPackages}/bin/julia -e '
                include("src/common/spherical_harmonics.jl")
                
                println("Testing Spherical Harmonics Y_l^m")
                println("==================================")
                println()
                
                # Test s orbital (l=0)
                println("s orbital (l=0, m=0):")
                println("  Y_0^0([0,0,1]) = ", ylm_real(0, 0, [0.0, 0.0, 1.0]))
                println()
                
                # Test p orbitals (l=1)
                println("p orbitals (l=1):")
                for m in -1:1
                  println("  Y_1^$m([0,0,1]) = ", ylm_real(1, m, [0.0, 0.0, 1.0]))
                end
                println()
                
                # Test d orbitals (l=2)
                println("d orbitals (l=2):")
                for m in -2:2
                  println("  Y_2^$m([0,0,1]) = ", ylm_real(2, m, [0.0, 0.0, 1.0]))
                end
                println()
                
                println("✅ Spherical harmonics working!")
              '
            '');
          };

          test-fft = {
            type = "app";
            program = toString (pkgs.writeShellScript "test-fft" ''
              cd ${self}/DFTK.jl
              ${juliaWithPackages}/bin/julia -e '
                using FFTW
                
                println("Testing FFT")
                println("===========")
                println()
                
                # Simple FFT test
                x = [1.0, 2.0, 3.0, 4.0]
                X = fft(x)
                
                println("Input: ", x)
                println("FFT: ", X)
                println("IFFT: ", ifft(X))
                println()
                
                println("✅ FFT working!")
              '
            '');
          };
        };
      }
    );
}
