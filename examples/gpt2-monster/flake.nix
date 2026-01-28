{
  description = "GPT-2 Monster Prime Resonance Analysis";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            cargo
            rustc
            rustfmt
            clippy
            pkg-config
            openssl
          ];
          
          shellHook = ''
            echo "🎪 GPT-2 Monster Prime Resonance Analysis"
            echo "========================================"
            echo ""
            echo "Run: cargo run --release"
            echo ""
            echo "This will:"
            echo "  1. Load GPT-2 (CPU inference)"
            echo "  2. Feed Monster primes as input"
            echo "  3. Trace all 12 layers"
            echo "  4. Detect prime resonances"
            echo "  5. Find harmonic patterns"
          '';
        };
      }
    );
}
