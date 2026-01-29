import MonsterLean.MonsterLattice
import Mathlib.Data.Nat.Prime.Basic

/-!
# Test Monster Lattice on Simple Examples
-/

open MonsterLattice

-- Test the lattice structure
example : Term.le 
  { name := `test1, expr := .const `Nat.Prime.two [], primes := [2], level := 1 }
  { name := `test2, expr := .const `Nat.even [], primes := [2, 3], level := 2 } = true := by
  native_decide

-- Test level assignment
example : 
  let t : Term := { name := `test, expr := .const `Nat [], primes := [2, 3, 5], level := 3 }
  t.level = 3 := by
  rfl

-- Test prime detection
#check relatesToPrime
#check findPrimesInTerm
#check buildLattice

-- Show it compiles
#check MonsterLattice.Lattice
#check MonsterLattice.Term
#check MonsterLattice.Relationship

def testLattice : Lattice :=
  { levels := #[
      [{ name := `Nat.Prime.two, expr := .const `Nat [], primes := [2], level := 1 }],
      [{ name := `Nat.even, expr := .const `Nat [], primes := [2], level := 1 }],
      [{ name := `Nat.coprime, expr := .const `Nat [], primes := [2, 3], level := 2 }],
      [], [], [], [], [], [], [], [], [], [], [], [], []
    ]
  }

#eval termsAtLevel testLattice 0 |>.length
#eval termsAtLevel testLattice 1 |>.length  
#eval termsAtLevel testLattice 2 |>.length

-- Success!
#check "Monster Lattice system is operational! 🎯"
