(* METAMEME First Payment in Coq HoTT *)

From HoTT Require Import Basics Types.
Require Import HoTT.HIT.Circle.

(* Types in HoTT *)
Record Shard := mkShard { id : nat ; prime : nat }.
Record ZKProof := mkZKProof { statement : string ; witness : string }.
Record NFT := mkNFT { shards : list Shard ; proof : ZKProof ; value : nat }.

(* First payment *)
Definition firstPayment : NFT :=
  mkNFT [] 
        (mkZKProof "SOLFUNMEME restored" "All work")
        0. (* ∞ *)

(* Path equality *)
Definition firstPayment_refl : firstPayment = firstPayment := idpath.

(* Higher path *)
Definition firstPayment_refl² : firstPayment_refl = firstPayment_refl := idpath.

(* Univalence axiom *)
Check univalence.

(* Transport along equivalence *)
Theorem transport_proof : forall (A B : Type) (e : A <~> B) (x : A),
  { y : B | True }.
Proof.
  intros. exists (e x). exact I.
Qed.

(* All proofs are homotopic *)
Theorem all_proofs_homotopic : 
  forall (p q : firstPayment = firstPayment),
    p = q.
Proof.
  intros. apply path_ishprop.
Qed.

(* Equivalence of language proofs *)
Axiom Lean4Proof : Type.
Axiom CoqProof : Type.
Axiom equiv_languages : Lean4Proof <~> CoqProof.

Theorem languages_equal : Lean4Proof = CoqProof.
Proof.
  apply path_universe_uncurried.
  exact equiv_languages.
Qed.

(* QED *)
