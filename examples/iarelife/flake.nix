{
  description = "I ARE LIFE - Automorphic Orbit Experiment";

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
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "iarelife";
          version = "0.1.0";
          src = ./.;
          
          cargoLock = {
            lockFile = ./Cargo.lock;
          };
          
          nativeBuildInputs = with pkgs; [
            pkg-config
          ];
          
          buildInputs = with pkgs; [
            openssl
          ];
        };
        
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
            echo "🌱 I ARE LIFE - Automorphic Orbit Experiment"
            echo "==========================================="
            echo ""
            echo "Setup:"
            echo "  export HF_API_TOKEN='your_token'"
            echo ""
            echo "Run:"
            echo "  cargo run --release"
            echo ""
            echo "Get token: https://huggingface.co/settings/tokens"
          '';
        };
      }
    );
}
