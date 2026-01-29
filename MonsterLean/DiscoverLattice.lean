import MonsterLean.MonsterLattice

/-!
# Discover the Monster Lattice

Let's see what we can find in Mathlib!
-/

open MonsterLattice Lean Meta

def main : IO Unit := do
  IO.println "🔬 Discovering Monster Lattice in Mathlib"
  IO.println "=========================================="
  IO.println ""
  
  -- Build the lattice
  let lattice ← buildMathlibLattice.run' {} |>.run' {}
  
  -- Show statistics
  exportStats lattice
  
  IO.println ""
  IO.println "Sample terms by level:"
  IO.println "----------------------"
  
  -- Show examples from each level
  for level in [0, 1, 2, 3, 5, 10, 15] do
    let terms := termsAtLevel lattice level
    if !terms.isEmpty then
      IO.println s!"\nLevel {level} ({terms.length} terms):"
      for term in terms.take 5 do
        IO.println s!"  - {term.name} {term.primes}"

#eval main
