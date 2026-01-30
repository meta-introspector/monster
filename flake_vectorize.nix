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

        # Pipelite job
        jobs.vectorize = {
          name = "vectorize-all-parquets";
          schedule = "0 * * * *"; # Hourly
          command = "${self.packages.${system}.default}/bin/vectorize_all_parquets";
          workdir = "/home/mdupont/experiments/monster";
        };
      }
    );
}
