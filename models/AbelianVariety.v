(* Coq model of Abelian variety over F_71 *)

Require Import QArith.
Require Import List.
Import ListNotations.

(* Abelian variety structure *)
Record AbelianVariety := {
  dimension : nat;
  fieldSize : nat;
  label : string;
  slopes : list Q
}.

(* The specific variety from LMFDB *)
Definition lmfdbVariety : AbelianVariety := {|
  dimension := 2;
  fieldSize := 71;
  label := "ah_a";
  slopes := [0 # 1; 1 # 2; 1 # 2; 1 # 1]
|}.

(* URL construction *)
Definition url (av : AbelianVariety) : string :=
  String.append "/Variety/Abelian/Fq/" 
    (String.append (string_of_nat av.(dimension))
      (String.append "/" 
        (String.append (string_of_nat av.(fieldSize))
          (String.append "/" av.(label))))).

(* Sum of slopes *)
Fixpoint sum_slopes (slopes : list Q) : Q :=
  match slopes with
  | [] => 0 # 1
  | s :: rest => Qplus s (sum_slopes rest)
  end.

(* Theorem: slopes sum to dimension *)
Theorem slopes_sum_to_dimension :
  sum_slopes lmfdbVariety.(slopes) == inject_Z (Z.of_nat lmfdbVariety.(dimension)).
Proof.
  unfold lmfdbVariety. simpl.
  unfold sum_slopes.
  (* 0 + 1/2 + 1/2 + 1 = 2 *)
  reflexivity.
Qed.

(* Theorem: URL is correct *)
Theorem url_correct :
  url lmfdbVariety = "/Variety/Abelian/Fq/2/71/ah_a".
Proof.
  unfold url, lmfdbVariety. simpl.
  reflexivity.
Qed.

(* Theorem: slopes match expected *)
Theorem slopes_correct :
  lmfdbVariety.(slopes) = [0 # 1; 1 # 2; 1 # 2; 1 # 1].
Proof.
  unfold lmfdbVariety. simpl.
  reflexivity.
Qed.
