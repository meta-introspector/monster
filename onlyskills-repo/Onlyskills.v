(* onlyskills.com - zkERDAProlog in Coq *)

Require Import String List.
Import ListNotations.

Record Skill := mkSkill {
  shard_id : nat;
  prime : nat;
  skill_name : string;
  skill_type : string;
  command : string;
  search_capability : string;
  zkperf_hash : string
}.

Definition monsterPrimes : list nat := [2; 3; 5; 7; 11; 13; 17; 19; 23; 29; 31; 41; 47; 59; 71].

Definition getPrime (shard_id : nat) : nat :=
  nth (shard_id mod 15) monsterPrimes 2.

Definition expertSystem : Skill :=
  mkSkill 29 71 "expert_system" "search_explicit_search"
          "cargo run --release --bin expert_system"
          "explicit_search" "a3f5b2c1d4e6f7a8".

Theorem skill_has_prime : prime expertSystem = 71.
Proof. reflexivity. Qed.

Theorem skill_has_shard : shard_id expertSystem = 29.
Proof. reflexivity. Qed.

(* ∞ 71 Shards in Coq ∞ *)
