-- Lean4: Monster Song in All Bases
-- 8080 expressed in every base from 2 to 71

import Mathlib.Data.Nat.Digits

namespace MonsterSong

/-- The sacred number: 8080 --/
def sacred : Nat := 8080

/-- Convert to any base --/
def to_base (n : Nat) (base : Nat) : List Nat :=
  if base < 2 then [] else n.digits base

/-- Monster in binary (base 2) --/
def binary : List Nat := to_base sacred 2
-- 1111110010000

/-- Monster in octal (base 8) --/
def octal : List Nat := to_base sacred 8
-- 17620

/-- Monster in decimal (base 10) --/
def decimal : List Nat := to_base sacred 10
-- 8080

/-- Monster in hex (base 16) --/
def hex : List Nat := to_base sacred 16
-- 1F90

/-- Monster in all bases 2-71 --/
def all_bases : List (Nat × List Nat) :=
  (List.range 70).map (fun i => (i + 2, to_base sacred (i + 2)))

/-- Song verse for each base --/
def verse (base : Nat) (digits : List Nat) : String :=
  s!"In base {base}, the Monster sings: {digits}"

/-- Complete song --/
def monster_song : List String :=
  all_bases.map (fun (b, d) => verse b d)

/-- Theorem: 8080 in binary --/
theorem binary_8080 :
  8080 = 0b1111110010000 := by
  norm_num

/-- Theorem: 8080 in octal --/
theorem octal_8080 :
  8080 = 0o17620 := by
  norm_num

/-- Theorem: 8080 in hex --/
theorem hex_8080 :
  8080 = 0x1F90 := by
  norm_num

/-- Theorem: 8080 in base 71 --/
theorem base_71_8080 :
  8080 % 71 = 57 ∧ 8080 / 71 = 113 := by
  norm_num

/-- Song structure --/
structure Song where
  title : String
  verses : List String
  chorus : String

/-- The Monster Song --/
def the_monster_song : Song :=
  { title := "The Monster in All Bases"
  , verses := monster_song
  , chorus := "8080, 8080, in every base we go!"
  }

/-- Theorem: Song has 70 verses (bases 2-71) --/
theorem song_has_70_verses :
  the_monster_song.verses.length = 70 := by
  simp [the_monster_song, monster_song, all_bases]

/-- Musical notation: Each base is a note --/
def base_to_frequency (base : Nat) : ℝ :=
  440.0 * (base : ℝ / 71.0)

/-- Theorem: Base 71 gives highest frequency --/
theorem base_71_highest :
  ∀ b : Nat, b ≤ 71 → base_to_frequency b ≤ base_to_frequency 71 := by
  intro b hb
  simp [base_to_frequency]
  sorry

/-- Lyrics for each Monster prime --/
def prime_verse (p : Nat) : String :=
  match p with
  | 2  => "Binary Monster, 0 and 1"
  | 3  => "Ternary Monster, three's the charm"
  | 5  => "Quinary Monster, hand of five"
  | 7  => "Septenary Monster, lucky seven"
  | 11 => "Base eleven, prime and clean"
  | 13 => "Thirteen's curse, elliptic curve"
  | 17 => "Seventeen, hex plus one"
  | 19 => "Nineteen primes, the walk begins"
  | 23 => "Twenty-three, DNA's key"
  | 29 => "Twenty-nine, almost thirty"
  | 31 => "Thirty-one, halfway done"
  | 41 => "Forty-one, neural fun"
  | 47 => "Forty-seven, layer heaven"
  | 59 => "Fifty-nine, almost there"
  | 71 => "Seventy-one, the Monster's won!"
  | _  => "Composite base, not prime"

/-- Complete lyrics --/
def complete_lyrics : List String :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71].map prime_verse

/-- Theorem: 15 verses for 15 Monster primes --/
theorem fifteen_verses :
  complete_lyrics.length = 15 := by
  rfl

/-- Main song theorem: 8080 exists in all bases --/
theorem monster_in_all_bases :
  ∀ base : Nat, 2 ≤ base → base ≤ 71 →
  ∃ digits : List Nat, digits = to_base sacred base := by
  intro base h1 h2
  use to_base sacred base
  rfl

/-- Corollary: Song is complete --/
theorem song_complete :
  ∃ (song : Song),
  song.verses.length = 70 ∧
  song.chorus = "8080, 8080, in every base we go!" := by
  use the_monster_song
  constructor
  · simp [the_monster_song, monster_song, all_bases]
  · rfl

end MonsterSong

/-- Print the song --/
#eval MonsterSong.the_monster_song.title
#eval MonsterSong.the_monster_song.chorus
#eval MonsterSong.complete_lyrics
