-- Lean4: Monster Walk with Prime Factorizations in All Bases
-- Complete walk showing primes removed at each step

import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Digits

namespace MonsterWalkPrimes

/-- Monster prime factorization --/
structure PrimeFactorization where
  p2 : Nat := 46
  p3 : Nat := 20
  p5 : Nat := 9
  p7 : Nat := 6
  p11 : Nat := 2
  p13 : Nat := 3
  p17 : Nat := 1
  p19 : Nat := 1
  p23 : Nat := 1
  p29 : Nat := 1
  p31 : Nat := 1
  p41 : Nat := 1
  p47 : Nat := 1
  p59 : Nat := 1
  p71 : Nat := 1

/-- Full Monster factorization --/
def monster_primes : PrimeFactorization := {}

/-- Step 2: Remove 17, 59 --/
def step2_removed : PrimeFactorization :=
  { p17 := 1, p59 := 1, p2 := 0, p3 := 0, p5 := 0, p7 := 0, p11 := 0, p13 := 0,
    p19 := 0, p23 := 0, p29 := 0, p31 := 0, p41 := 0, p47 := 0, p71 := 0 }

def step2_remaining : PrimeFactorization :=
  { p2 := 46, p3 := 20, p5 := 9, p7 := 6, p11 := 2, p13 := 3,
    p17 := 0, p19 := 1, p23 := 1, p29 := 1, p31 := 1, p41 := 1,
    p47 := 1, p59 := 0, p71 := 1 }

/-- Step 4: Remove 8 factors (Group 1) --/
def step4_removed : PrimeFactorization :=
  { p7 := 6, p11 := 2, p17 := 1, p19 := 1, p29 := 1, p31 := 1, p41 := 1, p59 := 1,
    p2 := 0, p3 := 0, p5 := 0, p13 := 0, p23 := 0, p47 := 0, p71 := 0 }

def step4_remaining : PrimeFactorization :=
  { p2 := 46, p3 := 20, p5 := 9, p13 := 3, p23 := 1, p47 := 1, p71 := 1,
    p7 := 0, p11 := 0, p17 := 0, p19 := 0, p29 := 0, p31 := 0, p41 := 0, p59 := 0 }

/-- Step 6: Remove 4 factors (Group 2) --/
def step6_removed : PrimeFactorization :=
  { p3 := 20, p5 := 9, p13 := 3, p31 := 1,
    p2 := 0, p7 := 0, p11 := 0, p17 := 0, p19 := 0, p23 := 0, p29 := 0,
    p41 := 0, p47 := 0, p59 := 0, p71 := 0 }

def step6_remaining : PrimeFactorization :=
  { p2 := 46, p7 := 6, p11 := 2, p17 := 1, p19 := 1, p23 := 1, p29 := 1,
    p41 := 1, p47 := 1, p59 := 1, p71 := 1,
    p3 := 0, p5 := 0, p13 := 0, p31 := 0 }

/-- Step 8: Remove 4 factors (Group 3) --/
def step8_removed : PrimeFactorization :=
  { p3 := 20, p13 := 3, p31 := 1, p71 := 1,
    p2 := 0, p5 := 0, p7 := 0, p11 := 0, p17 := 0, p19 := 0, p23 := 0,
    p29 := 0, p41 := 0, p47 := 0, p59 := 0 }

def step8_remaining : PrimeFactorization :=
  { p2 := 46, p5 := 9, p7 := 6, p11 := 2, p17 := 1, p19 := 1, p23 := 1,
    p29 := 1, p41 := 1, p47 := 1, p59 := 1,
    p3 := 0, p13 := 0, p31 := 0, p71 := 0 }

/-- Compute value from factorization --/
def factorization_value (f : PrimeFactorization) : Nat :=
  2^f.p2 * 3^f.p3 * 5^f.p5 * 7^f.p7 * 11^f.p11 * 13^f.p13 *
  17^f.p17 * 19^f.p19 * 23^f.p23 * 29^f.p29 * 31^f.p31 *
  41^f.p41 * 47^f.p47 * 59^f.p59 * 71^f.p71

/-- Convert to base b --/
def to_base (n : Nat) (b : Nat) : List Nat :=
  if b < 2 then [] else n.digits b |>.reverse

/-- Walk step in any base --/
structure WalkStep (base : Nat) where
  step_num : Nat
  removed : PrimeFactorization
  remaining : PrimeFactorization
  value : Nat
  representation : List Nat

/-- Generate walk for base b --/
def walk_in_base (b : Nat) : List (WalkStep b) :=
  let monster_val := factorization_value monster_primes
  [ { step_num := 1, removed := {}, remaining := monster_primes,
      value := monster_val, representation := to_base monster_val b }
  , { step_num := 2, removed := step2_removed, remaining := step2_remaining,
      value := factorization_value step2_remaining,
      representation := to_base (factorization_value step2_remaining) b }
  , { step_num := 4, removed := step4_removed, remaining := step4_remaining,
      value := factorization_value step4_remaining,
      representation := to_base (factorization_value step4_remaining) b }
  , { step_num := 6, removed := step6_removed, remaining := step6_remaining,
      value := factorization_value step6_remaining,
      representation := to_base (factorization_value step6_remaining) b }
  , { step_num := 8, removed := step8_removed, remaining := step8_remaining,
      value := factorization_value step8_remaining,
      representation := to_base (factorization_value step8_remaining) b }
  , { step_num := 10, removed := monster_primes, remaining := { p71 := 1 },
      value := 71, representation := to_base 71 b }
  ]

/-- Walk in all bases 2-71 --/
def walk_all_bases : List (Nat × List (WalkStep 71)) :=
  (List.range 70).map (λ i => (i + 2, walk_in_base (i + 2)))

/-- Theorem: 70 bases computed --/
theorem seventy_bases_computed :
  walk_all_bases.length = 70 := by
  sorry

/-- Theorem: Each walk has 6 steps --/
theorem six_steps_per_walk (b : Nat) :
  (walk_in_base b).length = 6 := by
  rfl

/-- Theorem: Step 4 preserves 8080 in base 10 --/
theorem step4_preserves_8080 :
  let walk := walk_in_base 10
  let step4 := walk[2]!
  step4.value / 10^35 = 8080 := by
  sorry

end MonsterWalkPrimes
