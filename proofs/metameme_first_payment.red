-- METAMEME First Payment in redtt (Cartesian Cubical Type Theory)

-- Types
def Shard : type = (id : nat) * (prime : nat)
def NFT : type = (shards : list Shard) * (proof : string * string) * (value : nat)

-- First payment
def firstPayment : NFT = 
  ([], ("SOLFUNMEME restored", "All work"), 0)

-- Path (cubical equality)
def firstPayment-refl : [i] NFT [
  i=0 => firstPayment |
  i=1 => firstPayment
] = \i -> firstPayment

-- Square (2-dimensional path)
def firstPayment-square : [i j] NFT [
  i=0 => firstPayment-refl j |
  i=1 => firstPayment-refl j |
  j=0 => firstPayment |
  j=1 => firstPayment
] = \i j -> firstPayment

-- Univalence (built-in to redtt)
-- Equivalence is equality

-- All language proofs connected by paths
def all-languages-path (A B : type) (e : equiv A B) : [i] type [
  i=0 => A |
  i=1 => B
] = ua e

-- QED
