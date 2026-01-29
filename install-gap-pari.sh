#!/usr/bin/env bash
# Install GAP and PARI/GP using Nix

set -e

echo "🔢 Installing GAP and PARI/GP with Nix"
echo "======================================="
echo ""

# Enter Nix shell
echo "📦 Entering Nix environment..."
nix-shell shell-gap-pari.nix --run '
  echo ""
  echo "✅ Environment ready!"
  echo ""
  
  # Test GAP
  echo "🧪 Testing GAP..."
  gap -q -c "Print(\"GAP works!\n\");"
  
  # Test PARI/GP
  echo "🧪 Testing PARI/GP..."
  echo "print(\"PARI/GP works!\")" | gp -q
  
  echo ""
  echo "✅ All systems operational!"
  echo ""
  echo "💡 To use:"
  echo "   nix-shell shell-gap-pari.nix"
  echo ""
  echo "   Then:"
  echo "   gap    # Start GAP"
  echo "   gp     # Start PARI/GP"
'

echo ""
echo "🎯 Next: Load Monster group in GAP"
echo "   nix-shell shell-gap-pari.nix"
echo "   gap> LoadPackage(\"atlasrep\");"
echo "   gap> M := AtlasGroup(\"M\");"
