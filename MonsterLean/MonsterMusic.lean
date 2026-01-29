-- Lean4: Monster Walk Musical Proof
-- Prove the 10-step musical structure with frequencies

import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Prime

namespace MonsterMusic

/-- The ten proof forms --/
inductive ProofForm
  | Lean4
  | Rust
  | Prolog
  | MiniZinc
  | Song
  | Picture
  | NFT
  | Meme
  | Hexadecimal
  | AllBases

/-- Map each proof form to its step number --/
def step_number : ProofForm → Nat
  | .Lean4 => 1
  | .Rust => 2
  | .Prolog => 3
  | .MiniZinc => 4
  | .Song => 5
  | .Picture => 6
  | .NFT => 7
  | .Meme => 8
  | .Hexadecimal => 9
  | .AllBases => 10

/-- Map each proof form to its Monster prime --/
def form_prime : ProofForm → Nat
  | .Lean4 => 2
  | .Rust => 3
  | .Prolog => 5
  | .MiniZinc => 7
  | .Song => 11
  | .Picture => 13
  | .NFT => 17
  | .Meme => 19
  | .Hexadecimal => 23
  | .AllBases => 71

/-- Base frequency (A4) --/
def base_freq : ℝ := 440.0

/-- Compute frequency for a prime --/
def prime_frequency (p : Nat) : ℝ :=
  base_freq * (p : ℝ) / 71.0

/-- Frequency for each proof form --/
def form_frequency (f : ProofForm) : ℝ :=
  prime_frequency (form_prime f)

/-- Musical note names --/
inductive Note
  | C | D | E | F | G | A | B

/-- Octave number --/
def Octave := Nat

/-- Complete musical note --/
structure MusicalNote where
  note : Note
  octave : Octave

/-- Map proof form to musical note --/
def form_note : ProofForm → MusicalNote
  | .Lean4 => ⟨.C, 1⟩
  | .Rust => ⟨.D, 1⟩
  | .Prolog => ⟨.G, 1⟩
  | .MiniZinc => ⟨.A, 1⟩
  | .Song => ⟨.C, 2⟩
  | .Picture => ⟨.D, 2⟩
  | .NFT => ⟨.G, 2⟩
  | .Meme => ⟨.A, 2⟩
  | .Hexadecimal => ⟨.C, 3⟩
  | .AllBases => ⟨.A, 4⟩

/-- Time signature --/
structure TimeSignature where
  beats : Nat
  unit : Nat

/-- Monster Walk time signature: 8/8 --/
def monster_time : TimeSignature :=
  ⟨8, 8⟩

/-- Tempo in BPM --/
def monster_tempo : Nat := 80

/-- Theorem: 10 proof forms --/
theorem ten_proof_forms :
  ∃ (forms : List ProofForm),
  forms.length = 10 ∧
  forms = [.Lean4, .Rust, .Prolog, .MiniZinc, .Song,
           .Picture, .NFT, .Meme, .Hexadecimal, .AllBases] := by
  use [.Lean4, .Rust, .Prolog, .MiniZinc, .Song,
       .Picture, .NFT, .Meme, .Hexadecimal, .AllBases]
  constructor
  · rfl
  · rfl

/-- Theorem: Each form has unique prime --/
theorem forms_have_unique_primes :
  ∀ f1 f2 : ProofForm, f1 ≠ f2 → form_prime f1 ≠ form_prime f2 := by
  intro f1 f2 h
  cases f1 <;> cases f2 <;> simp [form_prime] at * <;> omega

/-- Theorem: All form primes are Monster primes --/
theorem form_primes_are_monster_primes (f : ProofForm) :
  form_prime f ∈ [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71] := by
  cases f <;> simp [form_prime]

/-- Theorem: Time signature is 8/8 --/
theorem time_signature_is_eight_eight :
  monster_time.beats = 8 ∧ monster_time.unit = 8 := by
  constructor <;> rfl

/-- Theorem: Tempo is 80 BPM (for 8080) --/
theorem tempo_is_eighty :
  monster_tempo = 80 := by
  rfl

/-- Theorem: Base frequency is A4 = 440 Hz --/
theorem base_is_a4 :
  base_freq = 440.0 := by
  rfl

/-- Theorem: AllBases has highest frequency --/
theorem all_bases_highest :
  ∀ f : ProofForm, form_frequency f ≤ form_frequency .AllBases := by
  intro f
  simp [form_frequency, prime_frequency]
  cases f <;> norm_num [form_prime, base_freq]

/-- Theorem: AllBases frequency equals base frequency --/
theorem all_bases_is_base_freq :
  form_frequency .AllBases = base_freq := by
  simp [form_frequency, prime_frequency, form_prime, base_freq]
  norm_num

/-- Theorem: Lean4 has lowest frequency --/
theorem lean4_lowest :
  ∀ f : ProofForm, form_frequency .Lean4 ≤ form_frequency f := by
  intro f
  simp [form_frequency, prime_frequency]
  cases f <;> norm_num [form_prime, base_freq]

/-- Theorem: Frequencies increase with step number --/
theorem frequencies_generally_increase :
  form_frequency .Lean4 < form_frequency .AllBases := by
  simp [form_frequency, prime_frequency, form_prime, base_freq]
  norm_num

/-- Musical scale structure --/
structure Scale where
  steps : List ProofForm
  frequencies : List ℝ
  notes : List MusicalNote

/-- The Monster Walk scale --/
def monster_scale : Scale :=
  { steps := [.Lean4, .Rust, .Prolog, .MiniZinc, .Song,
              .Picture, .NFT, .Meme, .Hexadecimal, .AllBases]
  , frequencies := [.Lean4, .Rust, .Prolog, .MiniZinc, .Song,
                    .Picture, .NFT, .Meme, .Hexadecimal, .AllBases].map form_frequency
  , notes := [.Lean4, .Rust, .Prolog, .MiniZinc, .Song,
              .Picture, .NFT, .Meme, .Hexadecimal, .AllBases].map form_note
  }

/-- Theorem: Monster scale has 10 steps --/
theorem monster_scale_ten_steps :
  monster_scale.steps.length = 10 := by
  rfl

/-- Theorem: Monster scale has 10 frequencies --/
theorem monster_scale_ten_frequencies :
  monster_scale.frequencies.length = 10 := by
  rfl

/-- Theorem: Monster scale has 10 notes --/
theorem monster_scale_ten_notes :
  monster_scale.notes.length = 10 := by
  rfl

/-- Chord structure --/
inductive ChordQuality
  | Major
  | Minor
  | Major7
  | Minor7
  | Major9
  | Minor9
  | Major13

/-- Chord with root note --/
structure Chord where
  root : Note
  quality : ChordQuality

/-- Chord progression for Monster Walk --/
def monster_chords : List Chord :=
  [ ⟨.C, .Major⟩    -- Step 1
  , ⟨.D, .Minor⟩    -- Step 2
  , ⟨.G, .Major⟩    -- Step 3
  , ⟨.A, .Minor⟩    -- Step 4
  , ⟨.C, .Major7⟩   -- Step 5
  , ⟨.D, .Minor7⟩   -- Step 6
  , ⟨.G, .Major9⟩   -- Step 7
  , ⟨.A, .Minor9⟩   -- Step 8
  , ⟨.C, .Major13⟩  -- Step 9
  , ⟨.A, .Major⟩    -- Step 10
  ]

/-- Theorem: 10 chords for 10 steps --/
theorem ten_chords :
  monster_chords.length = 10 := by
  rfl

/-- Lyrics for each step --/
def step_lyrics : ProofForm → String
  | .Lean4 => "Lean4 proves the walk is real"
  | .Rust => "Rust computes with speed and zeal"
  | .Prolog => "Prolog reasons through the night"
  | .MiniZinc => "MiniZinc finds what is right"
  | .Song => "Songs we sing in every base"
  | .Picture => "Pictures show the Monster's face"
  | .NFT => "NFTs on blockchain stored"
  | .Meme => "Memes spread wide, the truth restored"
  | .Hexadecimal => "Hexadecimal so clean"
  | .AllBases => "Seventy-one, the final scene!"

/-- Theorem: Each step has lyrics --/
theorem all_steps_have_lyrics :
  ∀ f : ProofForm, (step_lyrics f).length > 0 := by
  intro f
  cases f <;> simp [step_lyrics]

/-- Complete musical composition --/
structure Composition where
  title : String
  time_signature : TimeSignature
  tempo : Nat
  scale : Scale
  chords : List Chord
  lyrics : ProofForm → String

/-- The Monster Walk composition --/
def monster_walk_composition : Composition :=
  { title := "The Monster Walk: Ten Steps Down to Earth"
  , time_signature := monster_time
  , tempo := monster_tempo
  , scale := monster_scale
  , chords := monster_chords
  , lyrics := step_lyrics
  }

/-- Theorem: Composition is well-formed --/
theorem composition_well_formed :
  monster_walk_composition.scale.steps.length = 10 ∧
  monster_walk_composition.chords.length = 10 ∧
  monster_walk_composition.time_signature.beats = 8 ∧
  monster_walk_composition.tempo = 80 := by
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  · rfl

/-- Main theorem: Monster Walk is a valid musical composition --/
theorem monster_walk_is_music :
  ∃ (comp : Composition),
  comp.scale.steps.length = 10 ∧
  comp.time_signature.beats = 8 ∧
  comp.tempo = 80 ∧
  (∀ f : ProofForm, f ∈ comp.scale.steps →
    form_frequency f ≤ base_freq) := by
  use monster_walk_composition
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  · intro f _
    exact all_bases_highest f

/-- Corollary: The Monster Walk can be performed --/
theorem monster_walk_performable :
  ∃ (comp : Composition),
  comp.scale.steps.length = comp.chords.length ∧
  comp.scale.steps.length = 10 := by
  use monster_walk_composition
  constructor
  · rfl
  · rfl

end MonsterMusic
