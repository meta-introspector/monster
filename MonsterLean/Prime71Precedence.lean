/-
# Prime 71 Precedence Theorems

Formal proofs about the structural significance of prime 71 as precedence
for graded ring multiplication in spectral sequence code.
-/

import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Nat.Factorization.Basic
import Mathlib.Algebra.Ring.Defs

namespace Prime71

/-- The 15 Monster primes -/
def MonsterPrimes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

/-- Precedence levels used in spectral library -/
def PrecedenceLevels : List Nat := [30, 35, 60, 65, 71, 73, 75, 78, 80]

/-- 71 is prime -/
theorem seventy_one_is_prime : Nat.Prime 71 := by norm_num

/-- 71 is in the Monster primes list -/
theorem seventy_one_in_monster : 71 ∈ MonsterPrimes := by decide

/-- 71 is the largest Monster prime -/
theorem seventy_one_is_largest_monster : 
  ∀ p ∈ MonsterPrimes, p ≤ 71 := by
  intro p hp
  fin_cases hp <;> norm_num

/-- 71 is in the precedence levels list -/
theorem seventy_one_in_precedence : 71 ∈ PrecedenceLevels := by decide

/-- 71 is the only Monster prime in precedence levels -/
theorem seventy_one_only_monster_precedence :
  ∀ n ∈ PrecedenceLevels, n ∈ MonsterPrimes → n = 71 := by
  intro n hn hm
  fin_cases hn <;> fin_cases hm <;> rfl

/-- 73 is prime but not a Monster prime -/
theorem seventy_three_prime_not_monster :
  Nat.Prime 73 ∧ 73 ∉ MonsterPrimes := by
  constructor
  · norm_num
  · decide

/-- Precedence 71 sits between 70 and 73 -/
theorem seventy_one_between_70_and_73 :
  70 < 71 ∧ 71 < 73 := by norm_num

/-- 72 is not in precedence levels (deliberately skipped) -/
theorem seventy_two_not_in_precedence :
  72 ∉ PrecedenceLevels := by decide

/-- 71 is the unique precedence level between 70 and 73 -/
theorem seventy_one_unique_in_range :
  ∀ n ∈ PrecedenceLevels, 70 < n ∧ n < 73 → n = 71 := by
  intro n hn ⟨h1, h2⟩
  interval_cases n
  · contradiction
  · rfl
  · contradiction

/-- Count of Monster primes dividing a number -/
def monsterDivisorCount (n : Nat) : Nat :=
  (MonsterPrimes.filter (· ∣ n)).length

/-- 71 has the minimal monster divisor count among precedence levels -/
theorem seventy_one_minimal_divisors :
  ∀ n ∈ PrecedenceLevels, n ≠ 73 → monsterDivisorCount 71 ≤ monsterDivisorCount n := by
  intro n hn h73
  fin_cases hn
  · -- 30 = 2 × 3 × 5, has 3 divisors
    norm_num [monsterDivisorCount]
  · -- 35 = 5 × 7, has 2 divisors
    norm_num [monsterDivisorCount]
  · -- 60 = 2² × 3 × 5, has 3 divisors
    norm_num [monsterDivisorCount]
  · -- 65 = 5 × 13, has 2 divisors
    norm_num [monsterDivisorCount]
  · -- 71 = 71, has 1 divisor (itself)
    rfl
  · -- 73 is excluded
    contradiction
  · -- 75 = 3 × 5², has 2 divisors
    norm_num [monsterDivisorCount]
  · -- 78 = 2 × 3 × 13, has 3 divisors
    norm_num [monsterDivisorCount]
  · -- 80 = 2⁴ × 5, has 2 divisors
    norm_num [monsterDivisorCount]

/-- 71 divides only itself among precedence levels -/
theorem seventy_one_divides_only_itself :
  ∀ n ∈ PrecedenceLevels, 71 ∣ n → n = 71 := by
  intro n hn hdiv
  fin_cases hn
  · -- 30: 71 ∤ 30
    norm_num at hdiv
  · -- 35: 71 ∤ 35
    norm_num at hdiv
  · -- 60: 71 ∤ 60
    norm_num at hdiv
  · -- 65: 71 ∤ 65
    norm_num at hdiv
  · -- 71: 71 ∣ 71
    rfl
  · -- 73: 71 ∤ 73
    norm_num at hdiv
  · -- 75: 71 ∤ 75
    norm_num at hdiv
  · -- 78: 71 ∤ 78
    norm_num at hdiv
  · -- 80: 71 ∤ 80
    norm_num at hdiv

/-- The gap from 70 to 71 is minimal (1) -/
theorem minimal_gap_from_70 :
  ∀ n ∈ PrecedenceLevels, 70 < n → 71 - 70 ≤ n - 70 := by
  intro n hn h70
  fin_cases hn
  · norm_num at h70
  · norm_num at h70
  · norm_num at h70
  · norm_num at h70
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · norm_num

/-- Structural theorem: 71 is uniquely positioned -/
theorem seventy_one_structural_position :
  71 ∈ PrecedenceLevels ∧
  71 ∈ MonsterPrimes ∧
  Nat.Prime 71 ∧
  (∀ p ∈ MonsterPrimes, p ≤ 71) ∧
  (∀ n ∈ PrecedenceLevels, n ∈ MonsterPrimes → n = 71) ∧
  (∀ n ∈ PrecedenceLevels, 70 < n ∧ n < 73 → n = 71) := by
  constructor
  · exact seventy_one_in_precedence
  constructor
  · exact seventy_one_in_monster
  constructor
  · exact seventy_one_is_prime
  constructor
  · exact seventy_one_is_largest_monster
  constructor
  · exact seventy_one_only_monster_precedence
  · exact seventy_one_unique_in_range

/-- Main theorem: 71 is the unique Monster prime precedence with minimal gap from 70 -/
theorem prime_71_uniqueness :
  ∃! n, n ∈ PrecedenceLevels ∧ 
        n ∈ MonsterPrimes ∧ 
        Nat.Prime n ∧
        70 < n ∧ n < 80 := by
  use 71
  constructor
  · constructor
    · exact seventy_one_in_precedence
    constructor
    · exact seventy_one_in_monster
    constructor
    · exact seventy_one_is_prime
    · norm_num
  · intro y ⟨hy_prec, hy_monster, hy_prime, hy_range⟩
    exact seventy_one_only_monster_precedence y hy_prec hy_monster

/-- Corollary: The choice of 71 is structurally determined -/
theorem choice_of_71_is_structural :
  ∀ n ∈ PrecedenceLevels,
    (n ∈ MonsterPrimes ∧ Nat.Prime n ∧ 70 < n ∧ n < 80) → n = 71 := by
  intro n hn ⟨hm, hp, hr⟩
  exact seventy_one_only_monster_precedence n hn hm

end Prime71
