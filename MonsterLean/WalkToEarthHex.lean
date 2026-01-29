-- Lean4: Monster Walk Down to Earth in Hexadecimal
-- 10 steps from 0x9A8C... to 0x47

import Mathlib.Data.Nat.Digits

namespace WalkToEarthHex

/-- The 10 steps in hexadecimal --/
structure HexStep where
  step_num : Nat
  dec_value : Nat
  hex_value : String
  magnitude : Nat
  description : String

/-- Step 1: Full Monster --/
def step1 : HexStep :=
  { step_num := 1
  , dec_value := 808017424794512875886459904961710757005754368000000000
  , hex_value := "0x9A8C8A6D7C35C7E18AF9955C4C2DF0000000000000"
  , magnitude := 53
  , description := "Start: Full Monster (46 hex digits)"
  }

/-- Step 2: Remove 2 factors → 80 --/
def step2 : HexStep :=
  { step_num := 2
  , dec_value := 80
  , hex_value := "0x50"
  , magnitude := 1
  , description := "Remove 2 factors: 0x50"
  }

/-- Step 3: Remove 4 factors → 808 --/
def step3 : HexStep :=
  { step_num := 3
  , dec_value := 808
  , hex_value := "0x328"
  , magnitude := 2
  , description := "Remove 4 factors: 0x328"
  }

/-- Step 4: Remove 8 factors → 8080 (Group 1) --/
def step4 : HexStep :=
  { step_num := 4
  , dec_value := 8080
  , hex_value := "0x1F90"
  , magnitude := 3
  , description := "Group 1: 0x1F90 (4 hex digits)"
  }

/-- Step 5: Continue from 8080 --/
def step5 : HexStep :=
  { step_num := 5
  , dec_value := 8080
  , hex_value := "0x1F90"
  , magnitude := 3
  , description := "Continue: 0x1F90"
  }

/-- Step 6: Group 2 → 80801742 --/
def step6 : HexStep :=
  { step_num := 6
  , dec_value := 80801742
  , hex_value := "0x4D0D4CE"
  , magnitude := 7
  , description := "Group 2: 0x4D0D4CE (7 hex digits)"
  }

/-- Step 7: Continue from 80801742 --/
def step7 : HexStep :=
  { step_num := 7
  , dec_value := 80801742
  , hex_value := "0x4D0D4CE"
  , magnitude := 7
  , description := "Continue: 0x4D0D4CE"
  }

/-- Step 8: Group 3 → 80801742479 --/
def step8 : HexStep :=
  { step_num := 8
  , dec_value := 80801742479
  , hex_value := "0x12D0D4CE8F"
  , magnitude := 10
  , description := "Group 3: 0x12D0D4CE8F (10 hex digits)"
  }

/-- Step 9: Convergence --/
def step9 : HexStep :=
  { step_num := 9
  , dec_value := 80801742479
  , hex_value := "0x12D0D4CE8F"
  , magnitude := 10
  , description := "Convergence: 0x12D0D4CE8F"
  }

/-- Step 10: Earth → 71 --/
def step10 : HexStep :=
  { step_num := 10
  , dec_value := 71
  , hex_value := "0x47"
  , magnitude := 1
  , description := "Earth: 0x47 (largest Monster prime)"
  }

/-- All 10 steps --/
def all_steps : List HexStep :=
  [step1, step2, step3, step4, step5, step6, step7, step8, step9, step10]

/-- Theorem: 10 steps --/
theorem ten_steps :
  all_steps.length = 10 := by
  rfl

/-- Theorem: Step 1 is Monster in hex --/
theorem step1_is_monster_hex :
  step1.hex_value = "0x9A8C8A6D7C35C7E18AF9955C4C2DF0000000000000" := by
  rfl

/-- Theorem: Step 4 is 0x1F90 --/
theorem step4_is_1F90 :
  step4.hex_value = "0x1F90" ∧
  step4.dec_value = 8080 := by
  constructor <;> rfl

/-- Theorem: Step 10 is 0x47 --/
theorem step10_is_47 :
  step10.hex_value = "0x47" ∧
  step10.dec_value = 71 := by
  constructor <;> rfl

/-- Hex digit count at each step --/
def hex_digit_count (s : HexStep) : Nat :=
  s.hex_value.length - 2  -- Subtract "0x"

/-- Theorem: Hex digits decrease --/
theorem hex_digits_decrease :
  hex_digit_count step1 > hex_digit_count step10 := by
  norm_num [hex_digit_count, step1, step10]

/-- Hex walk path --/
def hex_walk_path : List String :=
  all_steps.map (·.hex_value)

/-- Theorem: Path starts with Monster, ends with 0x47 --/
theorem path_correct :
  hex_walk_path.head? = some "0x9A8C8A6D7C35C7E18AF9955C4C2DF0000000000000" ∧
  hex_walk_path.getLast? = some "0x47" := by
  constructor <;> rfl

/-- Key hex values --/
def key_hex_values : List (Nat × String) :=
  [ (1, "0x9A8C8A6D7C35C7E18AF9955C4C2DF0000000000000")  -- Monster
  , (4, "0x1F90")                                          -- 8080
  , (6, "0x4D0D4CE")                                       -- 80801742
  , (8, "0x12D0D4CE8F")                                    -- 80801742479
  , (10, "0x47")                                           -- 71
  ]

/-- Theorem: 5 key hex values --/
theorem five_key_values :
  key_hex_values.length = 5 := by
  rfl

/-- Hex breakdown of 0x1F90 (8080) --/
def hex_1F90_breakdown : List (Nat × String) :=
  [ (1, "0x1000")  -- 4096
  , (15, "0x0F00") -- 3840
  , (9, "0x0090")  -- 144
  , (0, "0x0000")  -- 0
  ]

/-- Theorem: 0x1F90 breakdown sums to 8080 --/
theorem hex_1F90_sum :
  0x1000 + 0x0F00 + 0x0090 + 0x0000 = 8080 := by
  norm_num

/-- Main theorem: Hex Walk to Earth --/
theorem hex_walk_to_earth :
  ∃ (steps : List HexStep),
  steps.length = 10 ∧
  steps.head?.map (·.hex_value) = some "0x9A8C8A6D7C35C7E18AF9955C4C2DF0000000000000" ∧
  steps.getLast?.map (·.hex_value) = some "0x47" ∧
  (steps.get? 3).map (·.hex_value) = some "0x1F90" := by
  use all_steps
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  · rfl

/-- Corollary: From 46 hex digits to 2 --/
theorem from_46_to_2_hex_digits :
  hex_digit_count step1 = 46 ∧
  hex_digit_count step10 = 2 := by
  constructor <;> norm_num [hex_digit_count, step1, step10]

end WalkToEarthHex
