{
  description = "Monster Symmetry Search in Ollama Models";

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
            pkg-config
            openssl
          ];
          
          shellHook = ''
            echo "🎪 Monster Symmetry Search"
            echo "========================="
            echo ""
            echo "Model: qwen2.5:3b (1.9 GB)"
            echo "RAM: 30 GB available"
            echo ""
            echo "Search parameters:"
            echo "  - N-gram sizes: 4, 8, 16"
            echo "  - Depth: 10M, 5M, 2M"
            echo "  - Looking for: Monster primes, 8080, symmetries"
            echo ""
            echo "Run: cargo run --release"
          '';
        };
      }
    );
}
