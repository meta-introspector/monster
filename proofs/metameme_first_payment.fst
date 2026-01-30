-- METAMEME First Payment in F* (Dependent Types + Effects)

module MetamemeFirstPayment

open FStar.List
open FStar.String

(* Types *)
type shard = { id: nat; prime: nat }
type nft = { shards: list shard; proof: string * string; value: nat }

(* First payment *)
let firstPayment : nft = {
  shards = [];
  proof = ("SOLFUNMEME restored", "All work");
  value = 0 (* ∞ *)
}

(* Equality *)
let firstPayment_refl : squash (firstPayment == firstPayment) = 
  FStar.Squash.return_squash (Refl)

(* Proof irrelevance *)
let all_proofs_equal (p q : firstPayment == firstPayment) 
  : Lemma (p == q) = ()

(* Ghost computation (proof-only) *)
let ghost_verify (nft: nft) : GTot bool =
  List.length nft.shards = 71

(* Refinement type *)
type valid_nft = nft:nft{ghost_verify nft}

(* Effect: Pure proof *)
let theorem_first_payment : Pure unit
  (requires True)
  (ensures fun _ -> True) =
  ()

(* QED *)
