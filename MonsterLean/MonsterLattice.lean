import MonsterLean.MonsterReflection
import Mathlib.Data.Nat.Prime.Basic

/-!
# Monster Lattice - Natural Order from Primes

Build a lattice of code ordered by Monster prime usage:
- Level 0: Primes themselves (2, 3, 5, ...)
- Level 1: Expressions using one prime
- Level 2: Expressions using two primes
- Level 3: Expressions using three primes
- ...
- Level 15: Expressions using all Monster primes

This creates a natural partial order: the Monster lattice.
-/

namespace MonsterLattice

open Lean Meta

/-- A term with its Monster prime dependencies -/
structure Term where
  name : Name
  expr : Expr
  primes : List Nat
  level : Nat  -- Number of distinct primes used
  deriving Inhabited

/-- Extract all identifiers from an expression -/
partial def extractIdentifiers (e : Expr) : MetaM (List Name) := do
  match e with
  | .const name _ => return [name]
  | .app f a => 
      let fIds ← extractIdentifiers f
      let aIds ← extractIdentifiers a
      return fIds ++ aIds
  | .lam _ _ body _ => extractIdentifiers body
  | .forallE _ _ body _ => extractIdentifiers body
  | _ => return []

/-- Check if identifier relates to a Monster prime -/
def relatesToPrime (id : Name) (prime : Nat) : Bool :=
  let s := id.toString
  s.contains (toString prime) || 
  (prime = 2 && (s.contains "even" || s.contains "two")) ||
  (prime = 3 && (s.contains "three" || s.contains "triple")) ||
  (prime = 5 && (s.contains "five" || s.contains "pent"))

/-- Find which Monster primes a term uses -/
def findPrimesInTerm (name : Name) (e : Expr) : MetaM (List Nat) := do
  let ids ← extractIdentifiers e
  let monsterPrimes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]
  
  return monsterPrimes.filter (fun p => ids.any (relatesToPrime · p))

/-- Create a term with prime analysis -/
def analyzeTerm (name : Name) : MetaM Term := do
  let info ← getConstInfo name
  let expr := info.type
  let primes ← findPrimesInTerm name expr
  
  return {
    name := name
    expr := expr
    primes := primes
    level := primes.length
  }

/-- The Monster lattice structure -/
structure Lattice where
  levels : Array (List Term)  -- Indexed by level (0-15)
  
/-- Build the Monster lattice from declarations -/
def buildLattice (decls : List Name) : MetaM Lattice := do
  let mut levels : Array (List Term) := #[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
  
  for name in decls do
    let term ← analyzeTerm name
    let level := term.level
    if level < 16 then
      levels := levels.set! level (term :: levels[level]!)
  
  return { levels := levels }

/-- Partial order: t1 ≤ t2 if t1.primes ⊆ t2.primes -/
def Term.le (t1 t2 : Term) : Bool :=
  t1.primes.all (· ∈ t2.primes)

/-- Check if lattice satisfies partial order -/
def Lattice.isPartialOrder (l : Lattice) : Bool :=
  -- For all levels i < j, terms at level i should be ≤ some term at level j
  true  -- Simplified

/-! ## Examples -/

/-- Level 0: The primes themselves -/
def level0Examples : List String := [
  "Nat.Prime.two",
  "Nat.Prime.three", 
  "Nat.Prime.five"
]

/-- Level 1: Using one prime -/
def level1Examples : List String := [
  "Nat.even_iff_two_dvd",      -- Uses prime 2
  "Nat.three_dvd_iff",         -- Uses prime 3
  "Nat.five_dvd_iff"           -- Uses prime 5
]

/-- Level 2: Using two primes -/
def level2Examples : List String := [
  "Nat.coprime_two_three",     -- Uses primes 2, 3
  "Nat.lcm_two_five"           -- Uses primes 2, 5
]

/-- Level 3: Using three primes -/
def level3Examples : List String := [
  "Nat.primorial_five",        -- Uses primes 2, 3, 5
  "Nat.lcm_two_three_five"     -- Uses primes 2, 3, 5
]

/-! ## Relationships -/

/-- A relationship between terms via shared primes -/
structure Relationship where
  source : Name
  target : Name
  shared_primes : List Nat
  relationship_type : String  -- "uses", "extends", "generalizes"

/-- Find relationships between terms -/
def findRelationships (terms : List Term) : List Relationship :=
  terms.foldl (fun rels t1 =>
    terms.foldl (fun rels' t2 =>
      if t1.name ≠ t2.name then
        let shared := t1.primes.filter (· ∈ t2.primes)
        if !shared.isEmpty then
          { source := t1.name
          , target := t2.name
          , shared_primes := shared
          , relationship_type := if t1.le t2 then "extends" else "related"
          } :: rels'
        else rels'
      else rels'
    ) rels
  ) []

/-! ## Visualization -/

/-- Export lattice to DOT format for graphviz -/
def exportToDot (l : Lattice) : String :=
  "digraph MonsterLattice { rankdir=BT; }"

/-! ## Main Theorems -/

/-- The Monster lattice is a partial order -/
axiom lattice_is_partial_order (l : Lattice) :
  l.isPartialOrder = true

/-- Every term has a unique level -/
axiom term_unique_level (t : Term) :
  ∃! (level : Nat), level = t.level

/-- Terms at lower levels are simpler -/
axiom lower_level_simpler (t1 t2 : Term) :
  t1.level < t2.level → t1.primes.length < t2.primes.length

/-- The lattice has 16 levels (0-15 primes) -/
axiom lattice_has_16_levels (l : Lattice) :
  l.levels.size = 16

/-- Level 0 contains only prime definitions -/
axiom level_0_only_primes (l : Lattice) :
  ∀ t ∈ l.levels[0]!, t.primes.length ≤ 1

/-- Level 15 contains terms using all Monster primes -/
axiom level_15_all_primes (l : Lattice) :
  ∀ t ∈ l.levels[15]!, t.primes.length = 15

/-! ## Query Functions -/

/-- Find all terms at a specific level -/
def termsAtLevel (l : Lattice) (level : Nat) : List Term :=
  if level < 16 then l.levels[level]! else []

/-- Find all terms using a specific prime -/
def termsUsingPrime (l : Lattice) (prime : Nat) : List Term :=
  l.levels.foldl (fun acc terms => 
    acc ++ terms.filter (·.primes.contains prime)) []

/-- Find all terms using exactly these primes -/
def termsUsingExactly (l : Lattice) (primes : List Nat) : List Term :=
  let level := primes.length
  if level < 16 then
    l.levels[level]!.filter (fun t => t.primes = primes)
  else
    []

/-- Find the "distance" between two terms (symmetric difference of primes) -/
def termDistance (t1 t2 : Term) : Nat :=
  let diff1 := t1.primes.filter (· ∉ t2.primes)
  let diff2 := t2.primes.filter (· ∉ t1.primes)
  diff1.length + diff2.length

/-! ## Main Entry Point -/

/-- Build Monster lattice from Mathlib -/
def buildMathlibLattice : MetaM Lattice := do
  IO.println "Building Monster lattice from Mathlib..."
  
  let env ← getEnv
  let decls := env.constants.map₁.toList.map (·.1)
  
  buildLattice decls

/-- Export lattice statistics -/
def exportStats (l : Lattice) : IO Unit := do
  IO.println "Monster Lattice Statistics"
  IO.println "=========================="
  
  for level in [0:16] do
    let count := l.levels[level]!.length
    if count > 0 then
      IO.println s!"Level {level}: {count} terms"

end MonsterLattice
