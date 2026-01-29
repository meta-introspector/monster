#!/usr/bin/env bash

echo "🎯 MONSTER PRIME HISTOGRAM"
echo "=========================="
echo ""

# The 15 Monster Primes with emojis
declare -A PRIMES=(
  [2]="🌙 Binary Moon"
  [3]="🔺 Triangle"
  [5]="⭐ Pentagon"
  [7]="🎲 Heptagon"
  [11]="🎯 Hendecagon"
  [13]="💎 Tridecagon"
  [17]="🌊 Wave"
  [19]="🔮 Crystal"
  [23]="⚡ Lightning"
  [29]="🌀 Spiral"
  [31]="🔥 Fire"
  [41]="💫 Comet"
  [47]="🌟 Star"
  [59]="🌌 Galaxy"
  [71]="👹 Monster"
)

echo "Scanning Mathlib for Monster primes..."
echo ""

for prime in 2 3 5 7 11 13 17 19 23 29 31 41 47 59 71; do
  # Count occurrences
  COUNT=$(grep -r "\b$prime\b" .lake/packages/mathlib/Mathlib/Data/Nat/*.lean 2>/dev/null | wc -l)
  
  # Create bar
  BARS=""
  for ((i=0; i<COUNT/10; i++)); do
    BARS="${BARS}█"
  done
  
  # Get emoji
  EMOJI="${PRIMES[$prime]}"
  
  printf "%-3s %-20s %4d %s\n" "$prime" "$EMOJI" "$COUNT" "$BARS"
done

echo ""
echo "Legend:"
echo "  🌙 Binary Moon (2,3,5,7,11) - Most common"
echo "  🌊 Wave Crest (13,17,19,23,29) - Moderate"
echo "  👹 Deep Resonance (31,41,47,59,71) - Rare"

