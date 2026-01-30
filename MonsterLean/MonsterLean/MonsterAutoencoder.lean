-- Lean4 proof for create_monster_autoencoder.rs
-- Proves: Autoencoder architecture preserves dimensions

import Mathlib.Data.Matrix.Basic

-- Monster prime dimensions
def MonsterDims : List Nat := [5, 11, 23, 47, 71]

-- Layer transformation
structure Layer where
  input_dim : Nat
  output_dim : Nat

-- Encoder: 5 → 11 → 23 → 47 → 71
def encoder : List Layer := [
  ⟨5, 11⟩, ⟨11, 23⟩, ⟨23, 47⟩, ⟨47, 71⟩
]

-- Decoder: 71 → 47 → 23 → 11 → 5
def decoder : List Layer := [
  ⟨71, 47⟩, ⟨47, 23⟩, ⟨23, 11⟩, ⟨11, 5⟩
]

-- Composition preserves dimensions
def layers_compose (l1 l2 : Layer) : Prop :=
  l1.output_dim = l2.input_dim

-- Encoder is well-formed
theorem encoder_well_formed :
    ∀ i : Fin 3, layers_compose (encoder[i]!) (encoder[i.succ]!) := by
  intro i
  fin_cases i <;> rfl

-- Decoder is well-formed
theorem decoder_well_formed :
    ∀ i : Fin 3, layers_compose (decoder[i]!) (decoder[i.succ]!) := by
  intro i
  fin_cases i <;> rfl

-- Autoencoder reconstructs input dimension
theorem autoencoder_reconstructs :
    encoder.head!.input_dim = decoder.getLast!.output_dim := by
  rfl

#check encoder_well_formed
#check decoder_well_formed
#check autoencoder_reconstructs
