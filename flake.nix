{
  description = "Monster Group LMFDB Hecke Analysis";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        # Python environment for analysis
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          numpy
          scipy
          sympy
          pandas
          matplotlib
          jupyter
          huggingface-hub
        ]);
        
        # LMFDB database
        lmfdb-database = pkgs.stdenv.mkDerivation {
          name = "lmfdb-database";
          src = ./lmfdb-source;
          
          buildInputs = [ pkgs.postgresql pythonEnv ];
          
          buildPhase = ''
            # Initialize PostgreSQL
            initdb -D $out/data
            
            # Start PostgreSQL
            pg_ctl -D $out/data -l $out/logfile start
            
            # Wait for startup
            sleep 5
            
            # Create database
            createdb lmfdb
            
            # Load schema (if exists)
            if [ -f schema.sql ]; then
              psql lmfdb < schema.sql
            fi
            
            # Stop PostgreSQL
            pg_ctl -D $out/data stop
          '';
          
          installPhase = ''
            mkdir -p $out
            cp -r data $out/
          '';
        };
        
      in {
        packages = {
          inherit lmfdb-database;
          default = lmfdb-database;
        };
        
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.postgresql
            pkgs.perf
            pkgs.git
            pkgs.jq
            pkgs.act  # nektos/act for local GitHub Actions
          ];
          
          shellHook = ''
            echo "Monster Group LMFDB Analysis Environment"
            echo "========================================"
            echo ""
            echo "Available commands:"
            echo "  python3 analyze_lmfdb_source.py"
            echo "  python3 analyze_lmfdb_ast.py"
            echo "  python3 analyze_lmfdb_bytecode.py"
            echo "  ./trace_lmfdb_performance.sh"
            echo "  act -j analyze-lmfdb  # Run GitHub Actions locally"
            echo ""
          '';
        };
      }
    );
}
