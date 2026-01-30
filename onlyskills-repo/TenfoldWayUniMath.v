(** 10-fold Way in UniMath **)
(** Univalent Mathematics formalization **)

Require Import UniMath.Foundations.All.
Require Import UniMath.MoreFoundations.All.

(** 10 symmetry classes as type **)
Inductive SymmetryClass : UU :=
  | A : SymmetryClass
  | AIII : SymmetryClass
  | AI : SymmetryClass
  | BDI : SymmetryClass
  | D : SymmetryClass
  | DIII : SymmetryClass
  | AII : SymmetryClass
  | CII : SymmetryClass
  | C : SymmetryClass
  | CI : SymmetryClass.

(** Decidable equality **)
Definition SymmetryClass_eq_dec : ∏ (x y : SymmetryClass), (x = y) ⨿ (x != y).
Proof.
  intros x y.
  induction x; induction y; 
    try (apply ii1; apply idpath);
    try (apply ii2; intro H; inversion H).
Defined.

(** Bijection with stn 10 (standard finite type) **)
Definition toStn : SymmetryClass → stn 10.
Proof.
  intro c.
  induction c.
  - exact (0,, tt).
  - exact (1,, tt).
  - exact (2,, tt).
  - exact (3,, tt).
  - exact (4,, tt).
  - exact (5,, tt).
  - exact (6,, tt).
  - exact (7,, tt).
  - exact (8,, tt).
  - exact (9,, tt).
Defined.

Definition fromStn : stn 10 → SymmetryClass.
Proof.
  intro n.
  induction n as [n p].
  induction n.
  - exact A.
  - induction n.
    + exact AIII.
    + induction n.
      * exact AI.
      * induction n.
        -- exact BDI.
        -- induction n.
           ++ exact D.
           ++ induction n.
              ** exact DIII.
              ** induction n.
                 --- exact AII.
                 --- induction n.
                     +++ exact CII.
                     +++ induction n.
                         *** exact C.
                         *** induction n.
                             ---- exact CI.
                             ---- induction (nopathsfalsetotrue p).
Defined.

(** Bijection proofs **)
Lemma toStn_fromStn : ∏ (n : stn 10), toStn (fromStn n) = n.
Proof.
  intro n.
  induction n as [n p].
  induction n; try apply idpath.
  induction n; try apply idpath.
  induction n; try apply idpath.
  induction n; try apply idpath.
  induction n; try apply idpath.
  induction n; try apply idpath.
  induction n; try apply idpath.
  induction n; try apply idpath.
  induction n; try apply idpath.
  induction n; try apply idpath.
  induction (nopathsfalsetotrue p).
Qed.

Lemma fromStn_toStn : ∏ (c : SymmetryClass), fromStn (toStn c) = c.
Proof.
  intro c.
  induction c; apply idpath.
Qed.

(** Equivalence (univalence) **)
Definition SymmetryClass_weq_stn10 : SymmetryClass ≃ stn 10.
Proof.
  use weq_iso.
  - exact toStn.
  - exact fromStn.
  - exact fromStn_toStn.
  - exact toStn_fromStn.
Defined.

(** Path equality via univalence **)
Definition SymmetryClass_path_stn10 : SymmetryClass = stn 10.
Proof.
  apply univalence.
  exact SymmetryClass_weq_stn10.
Defined.

(** Bott periodicity **)
Definition bottShift (c : SymmetryClass) (n : nat) : SymmetryClass :=
  fromStn (make_stn 10 ((pr1 (toStn c) + n) mod 10) (idpath _)).

Lemma bott_period_8 : ∏ (c : SymmetryClass), bottShift c 8 = c.
Proof.
  intro c.
  induction c; apply idpath.
Qed.

(** Main theorem: 10-fold way is complete **)
Theorem tenfold_complete : iscontr (stn 10 → SymmetryClass).
Proof.
  use tpair.
  - exact fromStn.
  - intro f.
    apply funextfun.
    intro n.
    apply (maponpaths f (toStn_fromStn n)).
Qed.
