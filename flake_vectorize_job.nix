{
  description = "Monster vectorization pipeline";

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
          pname = "vectorize-all-parquets";
          version = "0.1.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;
          nativeBuildInputs = [ pkgs.pkg-config ];
          buildInputs = [ pkgs.openssl ];
        };

        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/vectorize_all_parquets";
        };

        jobs.vectorize = {
          name = "vectorize-all-parquets";
          command = "cd /home/mdupont/experiments/monster && nix develop --command bash -c 'cargo build --release --bin vectorize_all_parquets && ./target/release/vectorize_all_parquets'";
          logfile = "/home/mdupont/experiments/monster/logs/vectorize.log";
        };
      }
    );
}
