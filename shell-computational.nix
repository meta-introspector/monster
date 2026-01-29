{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "monster-computational-algebra";
  
  buildInputs = with pkgs; [
    # GAP - Groups, Algorithms, Programming
    gap
    
    # PARI/GP - Number theory
    pari
    
    # Optional: SageMath (includes both)
    # sagemath
    
    # Build tools
    git
    gcc
    gnumake
    autoconf
    automake
    libtool
    
    # For GAP packages
    gmp
    mpfr
    readline
    
    # For PARI
    gmp
    readline
  ];
  
  shellHook = ''
    echo "🔢 Computational Algebra Environment"
    echo "===================================="
    echo ""
    echo "✅ GAP: $(gap --version 2>&1 | head -1)"
    echo "✅ PARI/GP: $(gp --version 2>&1 | head -1)"
    echo ""
    echo "📚 GAP packages location: ${pkgs.gap}/share/gap/pkg"
    echo ""
    echo "💡 Usage:"
    echo "  gap          # Start GAP"
    echo "  gp           # Start PARI/GP"
    echo ""
    echo "🎯 Monster Group in GAP:"
    echo "  gap> LoadPackage(\"atlasrep\");"
    echo "  gap> M := AtlasGroup(\"M\");"
    echo "  gap> Order(M);"
  '';
}
