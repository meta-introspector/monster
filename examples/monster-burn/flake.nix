{
  description = "Monster Group Neural Network Construction with Burn";

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
            # GPU support (optional)
            vulkan-loader
            vulkan-headers
            vulkan-tools
          ];
          
          shellHook = ''
            echo "🎪 Monster Group Neural Network Construction"
            echo "============================================"
            echo ""
            echo "PROOF BY CONSTRUCTION:"
            echo "  Build 15 neural networks (one per Monster prime)"
            echo "  Index by Gödel numbers: G = p^p"
            echo "  Connect via Hecke operators: T_p = r_activation / r_weight"
            echo "  Verify lattice structure = Monster group"
            echo ""
            echo "Commands:"
            echo "  cargo run --release --bin prove-base-case"
            echo "  cargo run --release --bin prove-inductive"
            echo "  cargo run --release --bin construct-lattice"
            echo "  cargo run --release --bin verify-monster"
            echo ""
            echo "Features:"
            echo "  cargo run --features cuda  # CUDA backend"
            echo "  cargo run --features wgpu  # WebGPU backend (default)"
          '';
        };
      }
    );
}
