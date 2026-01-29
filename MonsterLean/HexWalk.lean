-- Lean4: The Hex Walk
-- Monster Walk through hexadecimal space: 0x1F90

import Mathlib.Data.Nat.Digits

namespace HexWalk

/-- Sacred number in decimal --/
def sacred_dec : Nat := 8080

/-- Sacred number in hex --/
def sacred_hex : Nat := 0x1F90

/-- Theorem: They are equal --/
theorem sacred_equality : sacred_dec = sacred_hex := by
  norm_num [sacred_dec, sacred_hex]

/-- Hex digits of 8080 --/
def hex_digits : List (Fin 16) := [1, 15, 9, 0]

/-- Hex digit names --/
def hex_name : Fin 16 → String
  | 0 => "0" | 1 => "1" | 2 => "2" | 3 => "3"
  | 4 => "4" | 5 => "5" | 6 => "6" | 7 => "7"
  | 8 => "8" | 9 => "9" | 10 => "A" | 11 => "B"
  | 12 => "C" | 13 => "D" | 14 => "E" | 15 => "F"

/-- The Hex Walk: 1 → F → 9 → 0 --/
inductive HexStep
  | One   -- 0x1000 = 4096
  | Eff   -- 0x0F00 = 3840
  | Nine  -- 0x0090 = 144
  | Zero  -- 0x0000 = 0

/-- Step values --/
def step_value : HexStep → Nat
  | .One => 0x1000
  | .Eff => 0x0F00
  | .Nine => 0x0090
  | .Zero => 0x0000

/-- Theorem: Sum equals 8080 --/
theorem hex_walk_sum :
  step_value .One + step_value .Eff + step_value .Nine + step_value .Zero = 8080 := by
  norm_num [step_value]

/-- Hex nibbles (4 bits each) --/
def nibble_1 : Nat := 0x1  -- 0001
def nibble_F : Nat := 0xF  -- 1111
def nibble_9 : Nat := 0x9  -- 1001
def nibble_0 : Nat := 0x0  -- 0000

/-- Theorem: Nibbles compose to 8080 --/
theorem nibbles_compose :
  nibble_1 * 16^3 + nibble_F * 16^2 + nibble_9 * 16^1 + nibble_0 * 16^0 = 8080 := by
  norm_num [nibble_1, nibble_F, nibble_9, nibble_0]

/-- Binary representation of each nibble --/
def nibble_binary : Fin 16 → List (Fin 2)
  | 0 => [0,0,0,0] | 1 => [0,0,0,1] | 2 => [0,0,1,0] | 3 => [0,0,1,1]
  | 4 => [0,1,0,0] | 5 => [0,1,0,1] | 6 => [0,1,1,0] | 7 => [0,1,1,1]
  | 8 => [1,0,0,0] | 9 => [1,0,0,1] | 10 => [1,0,1,0] | 11 => [1,0,1,1]
  | 12 => [1,1,0,0] | 13 => [1,1,0,1] | 14 => [1,1,1,0] | 15 => [1,1,1,1]

/-- The Hex Walk in binary --/
def hex_walk_binary : List (Fin 2) :=
  nibble_binary 1 ++ nibble_binary 15 ++ nibble_binary 9 ++ nibble_binary 0

/-- Theorem: Binary walk has 16 bits --/
theorem binary_walk_length :
  hex_walk_binary.length = 16 := by
  rfl

/-- Hex Walk through Monster primes --/
def prime_hex : Nat → Nat
  | 2 => 0x2
  | 3 => 0x3
  | 5 => 0x5
  | 7 => 0x7
  | 11 => 0xB
  | 13 => 0xD
  | 17 => 0x11
  | 19 => 0x13
  | 23 => 0x17
  | 29 => 0x1D
  | 31 => 0x1F
  | 41 => 0x29
  | 47 => 0x2F
  | 59 => 0x3B
  | 71 => 0x47
  | _ => 0

/-- Theorem: 71 in hex is 0x47 --/
theorem seventy_one_hex :
  prime_hex 71 = 0x47 := by
  rfl

/-- Hex Walk steps with Monster primes --/
structure HexWalkStep where
  digit : Fin 16
  value : Nat
  primes_below : List Nat

/-- Step 1: 0x1 --/
def step_1 : HexWalkStep :=
  { digit := 1
  , value := 0x1000
  , primes_below := []  -- No primes ≤ 1
  }

/-- Step 2: 0xF (15) --/
def step_2 : HexWalkStep :=
  { digit := 15
  , value := 0x0F00
  , primes_below := [2, 3, 5, 7, 11, 13]  -- 6 primes ≤ 15
  }

/-- Step 3: 0x9 (9) --/
def step_3 : HexWalkStep :=
  { digit := 9
  , value := 0x0090
  , primes_below := [2, 3, 5, 7]  -- 4 primes ≤ 9
  }

/-- Step 4: 0x0 --/
def step_4 : HexWalkStep :=
  { digit := 0
  , value := 0x0000
  , primes_below := []  -- No primes ≤ 0
  }

/-- The complete Hex Walk --/
def complete_hex_walk : List HexWalkStep :=
  [step_1, step_2, step_3, step_4]

/-- Theorem: 4 steps in Hex Walk --/
theorem four_hex_steps :
  complete_hex_walk.length = 4 := by
  rfl

/-- Hex Walk preserves 8080 --/
theorem hex_walk_preserves :
  (complete_hex_walk.map (·.value)).sum = 8080 := by
  norm_num [complete_hex_walk, step_1, step_2, step_3, step_4]

/-- Hex addresses (memory locations) --/
def hex_address : HexStep → Nat
  | .One => 0x1F90
  | .Eff => 0x0F90
  | .Nine => 0x0090
  | .Zero => 0x0000

/-- Theorem: Addresses decrease --/
theorem addresses_decrease :
  hex_address .One > hex_address .Eff ∧
  hex_address .Eff > hex_address .Nine ∧
  hex_address .Nine > hex_address .Zero := by
  norm_num [hex_address]

/-- Hex Walk as memory walk --/
def memory_walk : List Nat :=
  [0x1F90, 0x0F90, 0x0090, 0x0000]

/-- Theorem: Memory walk descends --/
theorem memory_descends :
  ∀ i : Fin 3, memory_walk[i]! > memory_walk[i.succ]! := by
  intro i
  fin_cases i <;> norm_num [memory_walk]

/-- Hex Walk through 71 shards --/
def shard_hex (n : Nat) : Nat := n % 71

/-- Theorem: 0x1F90 mod 71 --/
theorem hex_8080_shard :
  shard_hex 0x1F90 = 57 := by
  norm_num [shard_hex]

/-- Main theorem: The Hex Walk --/
theorem the_hex_walk :
  ∃ (steps : List Nat),
  steps = [0x1, 0xF, 0x9, 0x0] ∧
  steps.length = 4 ∧
  (steps.enum.map (fun (i, d) => d * 16^(3-i))).sum = 0x1F90 ∧
  0x1F90 = 8080 := by
  use [0x1, 0xF, 0x9, 0x0]
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · norm_num
  · norm_num

end HexWalk
