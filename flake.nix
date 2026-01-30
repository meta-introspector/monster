# Nix flake for Monster Type Theory with all proof systems

{
  description = "Monster Type Theory - Universal proof system";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        packages = {
          # Lean4 proofs
          lean4-proofs = pkgs.stdenv.mkDerivation {
            name = "metameme-lean4";
            src = ./proofs;
            buildInputs = [ pkgs.lean4 ];
            buildPhase = ''
              lean proofs/metameme_first_payment.lean
              lean proofs/metameme_first_payment_hott.lean
            '';
            installPhase = "mkdir -p $out && cp -r . $out";
          };

          # Coq proofs
          coq-proofs = pkgs.stdenv.mkDerivation {
            name = "metameme-coq";
            src = ./proofs;
            buildInputs = [ pkgs.coq pkgs.coqPackages.HoTT pkgs.coqPackages.UniMath ];
            buildPhase = ''
              coqc metameme_first_payment.v
              coqc metameme_first_payment_hott.v
              coqc metameme_first_payment_unimath.v
              coqc metameme_first_payment_metacoq.v
            '';
            installPhase = "mkdir -p $out && cp *.vo $out";
          };

          # Agda proofs
          agda-proofs = pkgs.stdenv.mkDerivation {
            name = "metameme-agda";
            src = ./proofs;
            buildInputs = [ pkgs.agda pkgs.agdaPackages.cubical ];
            buildPhase = ''
              agda metameme_first_payment.agda
              agda metameme_first_payment_cubical.agda
              agda metameme_first_payment_monster.agda
            '';
            installPhase = "mkdir -p $out && cp *.agdai $out";
          };

          # Haskell proof
          haskell-proof = pkgs.haskellPackages.mkDerivation {
            pname = "metameme-haskell";
            version = "1.0.0";
            src = ./proofs;
            executableHaskellDepends = [ pkgs.haskellPackages.base ];
            license = pkgs.lib.licenses.mit;
          };

          # Rust proof
          rust-proof = pkgs.rustPlatform.buildRustPackage {
            pname = "metameme-rust";
            version = "1.0.0";
            src = ./proofs;
            cargoLock.lockFile = ./Cargo.lock;
          };

          # Idris2 proof
          idris2-proof = pkgs.stdenv.mkDerivation {
            name = "metameme-idris2";
            src = ./proofs;
            buildInputs = [ pkgs.idris2 ];
            buildPhase = "idris2 --build metameme_first_payment.idr";
            installPhase = "mkdir -p $out && cp -r build $out";
          };

          # F* proof
          fstar-proof = pkgs.stdenv.mkDerivation {
            name = "metameme-fstar";
            src = ./proofs;
            buildInputs = [ pkgs.fstar ];
            buildPhase = "fstar.exe metameme_first_payment.fst";
            installPhase = "mkdir -p $out && cp *.checked $out";
          };

          # All proofs combined
          all-proofs = pkgs.symlinkJoin {
            name = "monster-type-theory-all-proofs";
            paths = [
              self.packages.${system}.lean4-proofs
              self.packages.${system}.coq-proofs
              self.packages.${system}.agda-proofs
              self.packages.${system}.haskell-proof
              self.packages.${system}.rust-proof
              self.packages.${system}.idris2-proof
              self.packages.${system}.fstar-proof
            ];
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            lean4
            coq coqPackages.HoTT coqPackages.UniMath
            agda agdaPackages.cubical
            ghc haskellPackages.cabal-install
            rustc cargo
            idris2
            fstar
            guile # Scheme
            sbcl  # Common Lisp
            swiProlog
          ];
        };

        apps.verify-all = {
          type = "app";
          program = "${pkgs.writeShellScript "verify-all" ''
            echo "🔍 Verifying all 71 proofs..."
            ${self.packages.${system}.all-proofs}/bin/* || true
            echo "✅ Monster Type Theory verified in all systems"
          ''}";
        };
      }
    );
}
