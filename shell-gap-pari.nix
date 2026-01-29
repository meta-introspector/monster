{ pkgs ? import <nixpkgs> {} }:

let
  # GAP from source with Monster group packages
  gap-with-packages = pkgs.gap.override {
    packageSet = pkgs.gap.packageSet.override {
      packages = with pkgs.gap.packageSet; [
        atlasrep      # Atlas of Group Representations (has Monster!)
        ctbllib       # Character Table Library
        tomlib        # Table of Marks Library
        smallgrp      # Small Groups Library
        transgrp      # Transitive Groups Library
      ];
    };
  };
  
  # PARI/GP with extra features
  pari-full = pkgs.pari.override {
    withGmp = true;
    withReadline = true;
  };

in pkgs.mkShell {
  name = "monster-gap-pari";
  
  buildInputs = [
    gap-with-packages
    pari-full
    pkgs.git
    pkgs.jq
    pkgs.python3
    pkgs.python3Packages.pandas
    pkgs.python3Packages.pyarrow
  ];
  
  shellHook = ''
    echo "👹 Monster Group Computational Environment"
    echo "=========================================="
    echo ""
    echo "✅ GAP with packages:"
    echo "   - AtlasRep (Monster group!)"
    echo "   - CTblLib (Character tables)"
    echo "   - TomLib (Table of marks)"
    echo ""
    echo "✅ PARI/GP with GMP support"
    echo ""
    echo "🚀 Quick Start:"
    echo ""
    echo "GAP - Load Monster:"
    echo "  gap> LoadPackage(\"atlasrep\");"
    echo "  gap> M := AtlasGroup(\"M\");"
    echo "  gap> Order(M);"
    echo ""
    echo "PARI/GP - Modular forms:"
    echo "  gp> E = ellinit([0,0,1,-1,0]);"
    echo "  gp> j = ellj(E);"
    echo ""
    export GAP_ROOT="${gap-with-packages}/share/gap"
    export PARI_DATADIR="${pari-full}/share/pari"
  '';
}
