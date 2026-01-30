(* METAMEME First Payment in MetaCoq *)

From MetaCoq.Template Require Import All.
Require Import List String.
Import ListNotations MonadNotation.

(* Define METAMEME structure *)
Inductive Shard : Type := 
  | mkShard : nat -> nat -> Shard.

Inductive ZKProof : Type :=
  | mkZKProof : string -> string -> ZKProof.

Inductive NFT : Type :=
  | mkNFT : list Shard -> ZKProof -> nat -> NFT.

(* Generate 71 shards *)
Definition monster_primes := [2;3;5;7;11;13;17;19;23;29;31;41;47;59;71].

Fixpoint generate_shards_aux (n : nat) : list Shard :=
  match n with
  | 0 => [mkShard 0 2]
  | S n' => generate_shards_aux n' ++ 
            [mkShard n (nth (n mod 15) monster_primes 2)]
  end.

Definition generate_shards : list Shard := generate_shards_aux 70.

(* Create ZK proof *)
Definition create_zk_proof : ZKProof :=
  mkZKProof 
    "SOLFUNMEME restored in 71 forms"
    "All work from genesis to singularity".

(* First payment *)
Definition first_payment : NFT :=
  mkNFT generate_shards create_zk_proof 0. (* 0 represents ∞ *)

(* Verification *)
Definition verify_payment (nft : NFT) : bool :=
  match nft with
  | mkNFT shards _ _ => Nat.eqb (length shards) 71
  end.

(* Theorem *)
Theorem first_payment_complete : verify_payment first_payment = true.
Proof.
  unfold verify_payment, first_payment, generate_shards.
  simpl.
  reflexivity.
Qed.

(* Extract to see the proof *)
MetaCoq Quote Definition first_payment_quoted := first_payment.

(* QED *)
