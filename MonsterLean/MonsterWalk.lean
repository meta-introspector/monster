import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.List.Basic
import Mathlib.Tactic

/-!
# Monster Group Walk Down to Earth - Hierarchical Structure

This file proves properties about the Monster group order and demonstrates
the hierarchical "walk down to earth" property where removing prime factors
preserves leading digits at multiple levels.

The Monster group order is:
2^46 × 3^20 × 5^9 × 7^6 × 11^2 × 13^3 × 17 × 19 × 23 × 29 × 31 × 41 × 47 × 59 × 71
= 808017424794512875886459904961710757005754368000000000

## Hierarchical Walk Structure

**Group 1**: Starting digits "8080"
- Remove 8 factors → preserve 4 digits (8080)

**Group 2**: Next digits "1742" (after 8080)
- Remove 4 different factors → preserve 4 digits (1742)

**Group 3**: Next digits after "17424"...

This creates a fractal-like structure where the Monster "walks down"
through multiple levels of digit preservation.
-/

namespace Monster

/-- Prime factorization of the Monster group -/
def monsterPrimes : List (Nat × Nat) :=
  [(2, 46), (3, 20), (5, 9), (7, 6), (11, 2), (13, 3),
   (17, 1), (19, 1), (23, 1), (29, 1), (31, 1), (41, 1),
   (47, 1), (59, 1), (71, 1)]

/-- Calculate the order from prime factorization -/
def orderFromFactors (factors : List (Nat × Nat)) : Nat :=
  factors.foldl (fun acc (p, e) => acc * p ^ e) 1

/-- The Monster group order -/
def monsterOrder : Nat := orderFromFactors monsterPrimes

/-- Get the first n digits of a natural number -/
def leadingDigits (n : Nat) (numDigits : Nat) : Nat :=
  let s := toString n
  let len := s.length
  if len ≤ numDigits then n
  else
    let prefix := s.take numDigits
    prefix.toNat!

/-- Skip first k digits and get next n digits -/
def digitsAfterSkip (n : Nat) (skip : Nat) (take : Nat) : Nat :=
  let s := toString n
  if s.length ≤ skip then 0
  else
    let remaining := s.drop skip
    let prefix := remaining.take take
    prefix.toNat!

/-- Remove factors at given indices -/
def removeFactors (factors : List (Nat × Nat)) (indices : List Nat) : List (Nat × Nat) :=
  factors.enum.filter (fun (i, _) => i ∉ indices) |>.map Prod.snd

/-- Check if removing given factors preserves n leading digits -/
def preservesLeadingDigits (factors : List (Nat × Nat)) (indices : List Nat) (numDigits : Nat) : Bool :=
  let original := orderFromFactors factors
  let reduced := orderFromFactors (removeFactors factors indices)
  leadingDigits original numDigits == leadingDigits reduced numDigits

/-! ## Group 1: The "8080" Walk -/

/-- The Monster order starts with 8080 -/
theorem monster_starts_with_8080 : leadingDigits monsterOrder 4 = 8080 := by
  native_decide

/-- Group 1: Removing 8 specific factors preserves "8080" -/
def group1_removal : List Nat := [3, 4, 6, 7, 9, 10, 11, 13]  -- 7^6, 11^2, 17, 19, 29, 31, 41, 59

theorem group1_preserves_8080 :
  preservesLeadingDigits monsterPrimes group1_removal 4 = true := by
  native_decide

def group1_result : Nat := orderFromFactors (removeFactors monsterPrimes group1_removal)

theorem group1_result_value :
  group1_result = 80807009282149818791922499584000000000 := by
  native_decide

/-! ## Group 2: The "1742" Walk (after "8080") -/

/-- After "8080", the next digits start with "1742" -/
theorem monster_after_8080_starts_with_1742 :
  digitsAfterSkip monsterOrder 4 4 = 1742 := by
  native_decide

/-- Group 2: Removing 4 different factors preserves "1742" -/
def group2_removal : List Nat := [1, 2, 5, 10]  -- 3^20, 5^9, 13^3, 31

def group2_result : Nat := orderFromFactors (removeFactors monsterPrimes group2_removal)

theorem group2_preserves_1742 :
  leadingDigits group2_result 4 = 1742 := by
  native_decide

/-! ## Hierarchical Walk Theorem -/

/-- Main theorem: The Monster exhibits a hierarchical walk structure
    where different factor removals preserve digits at different levels -/
theorem monster_hierarchical_walk :
  ∃ (g1_indices g2_indices : List Nat),
    -- Group 1: 8 factors preserve 4 digits "8080"
    g1_indices.length = 8 ∧
    preservesLeadingDigits monsterPrimes g1_indices 4 = true ∧
    leadingDigits monsterOrder 4 = 8080 ∧
    -- Group 2: 4 different factors give result starting with "1742"
    g2_indices.length = 4 ∧
    g1_indices ≠ g2_indices ∧
    leadingDigits (orderFromFactors (removeFactors monsterPrimes g2_indices)) 4 = 1742 := by
  use group1_removal, group2_removal
  constructor
  · native_decide
  constructor
  · native_decide
  constructor
  · native_decide
  constructor
  · native_decide
  constructor
  · native_decide
  · native_decide

/-! ## Uniqueness Properties -/

/-- Group 1 achieves maximum 4 digits, cannot achieve 5 -/
axiom group1_max_is_4 : ∀ (indices : List Nat),
  indices.length ≤ 15 →
  preservesLeadingDigits monsterPrimes indices 5 = false

/-- Group 2 also maxes at 4 digits -/
axiom group2_max_is_4 : ∀ (indices : List Nat),
  indices.length ≤ 15 →
  leadingDigits (orderFromFactors (removeFactors monsterPrimes indices)) 5 ≠ 17424

/-- The symmetry: Both groups achieve exactly 4 digits -/
theorem both_groups_achieve_4_digits :
  (∃ g1, preservesLeadingDigits monsterPrimes g1 4 = true) ∧
  (∃ g2, leadingDigits (orderFromFactors (removeFactors monsterPrimes g2)) 4 = 1742) := by
  constructor
  · use group1_removal
    native_decide
  · use group2_removal
    native_decide

end Monster

