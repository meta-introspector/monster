{ pkgs ? import <nixpkgs> {} }:

let
  # LMFDB server with 71-shard support
  lmfdb-server = pkgs.python3Packages.buildPythonApplication {
    pname = "lmfdb-71-shard";
    version = "0.1.0";
    
    src = ./.;
    
    propagatedBuildInputs = with pkgs.python3Packages; [
      flask
      pymongo
      sage
      pari
      numpy
      scipy
    ];
    
    meta = {
      description = "LMFDB server with 71-shard zkPrologML-ERDF support";
      license = pkgs.lib.licenses.agpl3;
    };
  };
  
  # zkPrologML-ERDF service
  zkprologml-erdf = pkgs.rustPlatform.buildRustPackage {
    pname = "zkprologml-erdf";
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
    
    meta = {
      description = "71-shard zkPrologML-ERDF service";
      license = pkgs.lib.licenses.agpl3;
    };
  };
  
  # 71-shard system
  shard-system = pkgs.writeShellScriptBin "lmfdb-71-shards" ''
    #!/usr/bin/env bash
    
    # Start LMFDB server
    ${lmfdb-server}/bin/lmfdb-server &
    LMFDB_PID=$!
    
    # Start 71 zkPrologML-ERDF shards
    for i in {0..70}; do
      ${zkprologml-erdf}/bin/zkprologml-erdf \
        --shard-id $i \
        --lmfdb-url http://localhost:5000 \
        --port $((8000 + i)) &
    done
    
    echo "✅ LMFDB server started (PID: $LMFDB_PID)"
    echo "✅ 71 zkPrologML-ERDF shards started (ports 8000-8070)"
    
    wait
  '';

in pkgs.buildEnv {
  name = "lmfdb-zkprologml-71-shards";
  paths = [
    lmfdb-server
    zkprologml-erdf
    shard-system
    pkgs.swiProlog
    pkgs.lean4
    pkgs.coq
  ];
}
