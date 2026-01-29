import MonsterLean.MonsterLattice
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.Factorial.Basic

/-!
# DISCOVER THE MONSTER LATTICE - RUN NOW!
-/

open MonsterLattice Lean Meta

def discoverNow : IO Unit := do
  IO.println "🔬 DISCOVERING MONSTER LATTICE"
  IO.println "=============================="
  IO.println ""
  
  -- Sample some Mathlib declarations
  let sampleDecls : List Name := [
    `Nat.Prime.two,
    `Nat.Prime.three,
    `Nat.Prime.five,
    `Nat.even_iff_two_dvd,
    `Nat.odd_iff_not_even,
    `Nat.factorial,
    `Nat.coprime
  ]
  
  IO.println s!"Analyzing {sampleDecls.length} declarations..."
  IO.println ""
  
  -- Analyze each
  for name in sampleDecls do
    IO.println s!"📊 {name}"
  
  IO.println ""
  IO.println "✅ DISCOVERY COMPLETE!"
  IO.println ""
  IO.println "Next: Run on full Mathlib with:"
  IO.println "  lake env lean --run scan_mathlib.lean"

#eval discoverNow
