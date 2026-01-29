import MonsterLean.MonsterReflection
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Algebra.Group.Defs

/-!
# Partition Mathlib by Monster Primes

Scan all Mathlib modules and partition by which Monster primes they use.
-/

namespace PartitionMathlib

open MonsterReflection Lean Meta

/-- Modules to scan -/
def mathlibModules : List Name := [
  `Mathlib.Data.Nat.Prime.Basic,
  `Mathlib.Data.Nat.Factorial.Basic,
  `Mathlib.Algebra.Group.Defs,
  `Mathlib.GroupTheory.Sylow,
  `Mathlib.NumberTheory.Divisors
]

/-- Scan a single module -/
def scanModule (modName : Name) : MetaM (List LatticePart) := do
  IO.println s!"Scanning {modName}..."
  
  let env ← getEnv
  let mut parts : List LatticePart := []
  
  -- Get all declarations in module
  for (name, _) in env.constants.map₁.toList do
    if name.getRoot == modName then
      -- Convert to JSON
      let json ← declToJson name
      
      -- Find Monster primes
      let primes := findMonsterPrimes json
      
      if !primes.isEmpty then
        IO.println s!"  {name}: primes {primes}"
        
        -- Add to partition
        for p in primes do
          parts := { prime := p, json_fragment := json, symmetry_count := 0 } :: parts
  
  return parts

/-- Scan all Mathlib modules -/
def scanAllMathlib : MetaM (List LatticePart) := do
  let mut allParts : List LatticePart := []
  
  for mod in mathlibModules do
    let parts ← scanModule mod
    allParts := allParts ++ parts
  
  return allParts

/-- Group partitions by prime -/
def groupByPrime (parts : List LatticePart) : List (Nat × Nat) :=
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]
  primes.map (fun p => (p, parts.filter (·.prime = p) |>.length))

/-- Main entry point -/
def main : IO Unit := do
  IO.println "🔬 Partitioning Mathlib by Monster Primes"
  IO.println "=========================================="
  IO.println ""
  
  let parts ← scanAllMathlib.run' {} |>.run' {}
  
  IO.println ""
  IO.println "Results:"
  IO.println "--------"
  
  let grouped := groupByPrime parts
  for (prime, count) in grouped do
    if count > 0 then
      IO.println s!"Prime {prime}: {count} declarations"
  
  IO.println ""
  IO.println s!"Total: {parts.length} declarations partitioned"

end PartitionMathlib
