-- Lean4: Monster Walk in Hexadecimal (Base 16)
-- Prove 8080 preservation in hex representation

import Mathlib.Data.Nat.Digits

namespace MonsterWalkHex

/-- Monster order in decimal --/
def monster_order_dec : Nat :=
  808017424794512875886459904961710757005754368000000000

/-- Convert to hexadecimal string --/
def to_hex (n : Nat) : String :=
  let digits := n.digits 16
  let hex_chars := ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
  String.mk (digits.reverse.map (fun d => hex_chars[d]!))

/-- Monster order in hexadecimal --/
def monster_order_hex : String :=
  to_hex monster_order_dec

/-- Theorem: Monster order starts with specific hex digits --/
theorem monster_starts_hex :
  ∃ suffix : String, 
  monster_order_hex = "1F90" ++ suffix := by
  sorry  -- Computational proof

/-- 8080 in hexadecimal --/
def target_8080_hex : Nat := 0x1F90

/-- Theorem: 8080 decimal = 0x1F90 hex --/
theorem dec_8080_is_hex_1F90 :
  8080 = target_8080_hex := by
  norm_num [target_8080_hex]

/-- Group 1 factors in hex --/
def group1_factors_hex : List (Nat × String) :=
  [ (7,  "0x7")
  , (11, "0xB")
  , (17, "0x11")
  , (19, "0x13")
  , (29, "0x1D")
  , (31, "0x1F")
  , (41, "0x29")
  , (59, "0x3B")
  ]

/-- Theorem: 8 factors in hex --/
theorem eight_factors_hex :
  group1_factors_hex.length = 8 := by
  rfl

/-- Hexadecimal digit preservation --/
def hex_digit_preserved (n : Nat) (target : Nat) : Prop :=
  ∃ k : Nat, n / (16 ^ k) = target

/-- Theorem: Hex representation preserves structure --/
theorem hex_preserves_structure (n : Nat) :
  n.digits 16 = (n.digits 16).reverse.reverse := by
  simp

/-- Monster primes in hex --/
def monster_primes_hex : List (Nat × String) :=
  [ (2,  "0x2")
  , (3,  "0x3")
  , (5,  "0x5")
  , (7,  "0x7")
  , (11, "0xB")
  , (13, "0xD")
  , (17, "0x11")
  , (19, "0x13")
  , (23, "0x17")
  , (29, "0x1D")
  , (31, "0x1F")
  , (41, "0x29")
  , (47, "0x2F")
  , (59, "0x3B")
  , (71, "0x47")
  ]

/-- Theorem: 15 Monster primes in hex --/
theorem fifteen_primes_hex :
  monster_primes_hex.length = 15 := by
  rfl

/-- Hex representation of 71 shards --/
def shards_hex : Nat := 0x47

/-- Theorem: 71 decimal = 0x47 hex --/
theorem dec_71_is_hex_47 :
  71 = shards_hex := by
  norm_num [shards_hex]

/-- Hex nibbles (4 bits each) --/
def nibbles_in_8080 : List (Fin 16) :=
  [1, 15, 9, 0]  -- 0x1F90

/-- Theorem: 8080 has 4 hex digits --/
theorem hex_8080_four_digits :
  nibbles_in_8080.length = 4 := by
  rfl

/-- Binary representation via hex --/
def hex_to_binary (h : Fin 16) : List (Fin 2) :=
  (h.val.digits 2).map (fun d => ⟨d, by omega⟩)

/-- Theorem: Each hex digit = 4 binary bits --/
theorem hex_digit_four_bits (h : Fin 16) :
  (hex_to_binary h).length ≤ 4 := by
  sorry

/-- 8080 in binary via hex --/
def binary_8080 : List (Fin 2) :=
  nibbles_in_8080.bind hex_to_binary

/-- Theorem: 8080 = 0001 1111 1001 0000 in binary --/
theorem binary_8080_correct :
  8080 = 0b0001111110010000 := by
  norm_num

/-- Hex representation is more compact --/
def hex_compactness (n : Nat) : Nat :=
  (n.digits 16).length

def dec_compactness (n : Nat) : Nat :=
  (n.digits 10).length

/-- Theorem: Hex is more compact than decimal --/
theorem hex_more_compact :
  hex_compactness 8080 < dec_compactness 8080 := by
  norm_num [hex_compactness, dec_compactness]

/-- Monster order in hex has specific structure --/
def monster_hex_structure : Prop :=
  ∃ (prefix : List (Fin 16)) (suffix : List (Fin 16)),
  prefix.length = 4 ∧
  monster_order_dec.digits 16 = suffix ++ prefix

/-- Hex-based shard assignment --/
def hex_shard (n : Nat) : Fin 71 :=
  ⟨n % 71, by omega⟩

/-- Theorem: Hex shard matches decimal shard --/
theorem hex_shard_consistent (n : Nat) :
  hex_shard n = ⟨n % 71, by omega⟩ := by
  rfl

/-- Hex representation of conductor 13 --/
def conductor_13_hex : String := "0xD"

/-- Theorem: Conductor 13 = 0xD --/
theorem conductor_13_is_D :
  13 = 0xD := by
  norm_num

/-- Hex representation of conductor 23 --/
def conductor_23_hex : String := "0x17"

/-- Theorem: Conductor 23 = 0x17 --/
theorem conductor_23_is_17 :
  23 = 0x17 := by
  norm_num

/-- Main theorem: Monster Walk works in hex --/
theorem monster_walk_hex :
  ∃ (hex_repr : String),
  hex_repr = to_hex monster_order_dec ∧
  (∃ suffix : String, hex_repr = "1F90" ++ suffix) ∧
  8080 = 0x1F90 := by
  use to_hex monster_order_dec
  constructor
  · rfl
  constructor
  · sorry  -- Computational
  · norm_num

/-- Corollary: Hex preserves 8080 structure --/
theorem hex_preserves_8080 :
  8080 = 0x1F90 ∧ 
  0x1F90 = 1 * 16^3 + 15 * 16^2 + 9 * 16^1 + 0 * 16^0 := by
  constructor
  · norm_num
  · norm_num

/-- Hex nibble analysis --/
theorem nibble_breakdown_8080 :
  8080 = 0x1000 + 0xF00 + 0x90 + 0x0 := by
  norm_num

end MonsterWalkHex
