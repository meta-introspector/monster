#!/usr/bin/env bash

echo "🔍 SEARCHING ALL 15 MONSTER PRIMES IN MATHLIB"
echo "=============================================="
echo ""

for prime in 2 3 5 7 11 13 17 19 23 29 31 41 47 59 71; do
  COUNT=$(grep -r "\b$prime\b" .lake/packages/mathlib/Mathlib --include="*.lean" 2>/dev/null | wc -l)
  
  # Emoji
  case $prime in
    2) EMOJI="🌙";;
    3) EMOJI="🔺";;
    5) EMOJI="⭐";;
    7) EMOJI="🎲";;
    11) EMOJI="🎯";;
    13) EMOJI="💎";;
    17) EMOJI="🌊";;
    19) EMOJI="🔮";;
    23) EMOJI="⚡";;
    29) EMOJI="🌀";;
    31) EMOJI="🔥";;
    41) EMOJI="💫";;
    47) EMOJI="🌟";;
    59) EMOJI="🌌";;
    71) EMOJI="👹";;
  esac
  
  # Bar
  BARS=""
  for ((i=0; i<COUNT/100; i++)); do
    BARS="${BARS}█"
  done
  
  printf "%2s %s %5d %s\n" "$prime" "$EMOJI" "$COUNT" "$BARS"
done

echo ""
echo "✅ Complete scan of all Mathlib!"

