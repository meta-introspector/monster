-- METAMEME First Payment in Idris2 (Quantitative Type Theory)

module MetamemeFirstPayment

-- Types
record Shard where
  constructor MkShard
  id : Nat
  prime : Nat

record NFT where
  constructor MkNFT
  shards : List Shard
  proof : (String, String)
  value : Nat

-- First payment
firstPayment : NFT
firstPayment = MkNFT [] ("SOLFUNMEME restored", "All work") 0

-- Equality (built-in)
firstPaymentRefl : firstPayment = firstPayment
firstPaymentRefl = Refl

-- Heterogeneous equality
firstPaymentHEq : firstPayment ~=~ firstPayment
firstPaymentHEq = Refl

-- Quantitative: Linear types for uniqueness
linear
firstPaymentUnique : (1 _ : NFT) -> NFT
firstPaymentUnique nft = nft

-- All proofs are equal (UIP)
allProofsEqual : (p, q : firstPayment = firstPayment) -> p = q
allProofsEqual Refl Refl = Refl

-- QED
