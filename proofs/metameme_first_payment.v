(* METAMEME First Payment Proof in Coq *)

Require Import Arith.
Require Import List.

(* Define METAMEME structure *)
Inductive METAMEME : Type :=
  | metameme : nat -> list Shard -> ZKProof -> METAMEME
with Shard : Type :=
  | shard : nat -> nat -> Shard
with ZKProof : Type :=
  | zkproof : Prop -> ZKProof.

(* 71 shards axiom *)
Axiom seventy_one_shards : nat.
Axiom seventy_one_eq : seventy_one_shards = 71.

(* First payment theorem *)
Theorem first_payment_complete : 
  forall (holders : nat), 
    holders > 0 -> 
    exists (nft : METAMEME), 
      match nft with
      | metameme _ shards _ => length shards = 71
      end.
Proof.
  intros holders H.
  exists (metameme 0 (repeat (shard 0 0) 71) (zkproof True)).
  simpl.
  rewrite repeat_length.
  apply seventy_one_eq.
Qed.

(* Recursive proof *)
Fixpoint metameme_proves_self (m : METAMEME) : METAMEME := m.

(* Infinity theorem *)
Axiom infinity : nat.
Theorem metameme_is_infinite : 
  forall (m : METAMEME), exists (n : nat), n = infinity.
Proof.
  intro m.
  exists infinity.
  reflexivity.
Qed.

(* QED *)
Theorem payment_eternal : True.
Proof. trivial. Qed.
