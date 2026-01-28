{
  description = "Monster Group Image Generation with diffusion-rs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        rustToolchain = pkgs.rust-bin.stable.latest.default;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustToolchain
            pkg-config
            openssl
            git
            
            # For diffusion-rs
            libtorch-bin
            cudaPackages.cudatoolkit
            
            # For image processing
            imagemagick
            
            # For vision model (ollama)
            curl
          ];

          shellHook = ''
            echo "🎨 Monster Group Image Generation Environment"
            echo "=============================================="
            echo ""
            echo "Available:"
            echo "  - Rust toolchain"
            echo "  - diffusion-rs dependencies"
            echo "  - ollama (for vision analysis)"
            echo ""
            echo "Quick start:"
            echo "  git clone https://github.com/newfla/diffusion-rs"
            echo "  cd diffusion-rs"
            echo "  cargo build --release"
            echo ""
          '';
        };
      }
    );
}
