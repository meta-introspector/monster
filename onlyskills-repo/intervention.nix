{ pkgs ? import <nixpkgs> {} }:

pkgs.stdenv.mkDerivation {
  pname = "intervention-tools";
  version = "1.0.0";
  
  src = ./.;
  
  buildInputs = with pkgs; [
    lean4
    rustc
    cargo
    python3
    linuxPackages.perf
  ];
  
  buildPhase = ''
    # Build Lean4 proof
    lean intervention_proof.lean
    
    # Build Rust analyzer
    rustc process_analyzer.rs -o process_analyzer
    
    # Verify Python tools
    python3 -m py_compile intervention.py
  '';
  
  installPhase = ''
    mkdir -p $out/bin
    mkdir -p $out/proofs
    
    # Install binaries
    cp process_analyzer $out/bin/
    cp intervention.py $out/bin/
    
    # Install proofs
    cp intervention_proof.lean $out/proofs/
    cp intervention_receipt.json $out/proofs/
  '';
  
  meta = with pkgs.lib; {
    description = "Tools to intervene in searching processes";
    license = licenses.mit;
    platforms = platforms.linux;
  };
}
