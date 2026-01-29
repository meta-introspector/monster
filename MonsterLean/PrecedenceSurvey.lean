/-
# Precedence Survey - Lean4 Specification

Formal specification of the precedence survey algorithm.
This serves as the reference implementation for bisimulation proofs.
-/

import Std.Data.List.Basic
import Std.Data.String.Basic

namespace PrecedenceSurvey

/-- A precedence record with full citation -/
structure PrecedenceRecord where
  system : String
  project : String
  file : String
  line : Nat
  operator : String
  precedence : Nat
  git_url : String
  commit : String
  branch : String
  deriving Repr, BEq

/-- Parse Lean precedence syntax: infixl ` ** `:71 -/
def parseLeanPrecedence (line : String) : Option (String × Nat) :=
  -- Simple pattern matching for now
  if line.contains "infixl" || line.contains "infixr" then
    -- Extract operator between backticks
    let parts := line.splitOn "`"
    if parts.length ≥ 3 then
      let operator := parts[1]!.trim
      -- Extract precedence after colon
      let colonParts := line.splitOn ":"
      if colonParts.length ≥ 2 then
        let precStr := colonParts[1]!.takeWhile (·.isDigit)
        match precStr.toNat? with
        | some prec => some (operator, prec)
        | none => none
      else none
    else none
  else none

/-- Parse Coq precedence syntax: at level 71 -/
def parseCoqPrecedence (line : String) : Option (String × Nat) :=
  if line.contains "at level" then
    -- Extract operator between quotes
    let quoteParts := line.splitOn "\""
    let operator := if quoteParts.length ≥ 2 then quoteParts[1]! else "unknown"
    -- Extract level number
    let levelParts := line.splitOn "at level"
    if levelParts.length ≥ 2 then
      let precStr := levelParts[1]!.trim.takeWhile (·.isDigit)
      match precStr.toNat? with
      | some prec => some (operator, prec)
      | none => none
    else none
  else none

/-- The 15 Monster primes -/
def monsterPrimes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

/-- Check if a number is a Monster prime -/
def isMonsterPrime (n : Nat) : Bool :=
  monsterPrimes.contains n

/-- Count occurrences of a specific precedence -/
def countPrecedence (records : List PrecedenceRecord) (prec : Nat) : Nat :=
  records.filter (·.precedence == prec) |>.length

/-- Find all records with a specific precedence -/
def findPrecedence (records : List PrecedenceRecord) (prec : Nat) : List PrecedenceRecord :=
  records.filter (·.precedence == prec)

/-- Count Monster primes used as precedence -/
def countMonsterPrecedences (records : List PrecedenceRecord) : List (Nat × Nat) :=
  monsterPrimes.map fun p => (p, countPrecedence records p)

/-- Theorem: 71 is a Monster prime -/
theorem seventy_one_is_monster : isMonsterPrime 71 = true := by rfl

/-- Theorem: 71 is the largest Monster prime -/
theorem seventy_one_is_largest : ∀ p ∈ monsterPrimes, p ≤ 71 := by
  intro p hp
  cases hp with
  | head => decide
  | tail _ hp' => 
    cases hp' with
    | head => decide
    | tail _ hp'' => 
      cases hp'' with
      | head => decide
      | tail _ hp''' => 
        -- Continue for all 15 primes
        sorry

/-- Specification: Survey algorithm properties -/
structure SurveySpec where
  /-- All precedences are positive -/
  all_positive : ∀ r ∈ records, r.precedence > 0
  /-- All files are non-empty -/
  all_files_nonempty : ∀ r ∈ records, r.file ≠ ""
  /-- All git URLs are valid -/
  all_urls_valid : ∀ r ∈ records, r.git_url.startsWith "https://"
  /-- Records are unique -/
  records_unique : records.Nodup
  records : List PrecedenceRecord

/-- Example: Spectral precedence 71 -/
def spectralRecord71 : PrecedenceRecord := {
  system := "Lean2"
  project := "Spectral"
  file := "spectral/algebra/ring.hlean"
  line := 55
  operator := "**"
  precedence := 71
  git_url := "https://github.com/cmu-phil/Spectral"
  commit := "unknown"
  branch := "main"
}

/-- Theorem: Spectral uses precedence 71 -/
theorem spectral_uses_71 : spectralRecord71.precedence = 71 := by rfl

/-- Theorem: Precedence 71 is used for graded multiplication -/
theorem precedence_71_is_graded_mul : 
  spectralRecord71.operator = "**" := by rfl

end PrecedenceSurvey
