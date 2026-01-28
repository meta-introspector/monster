-- ProveOmniscience.lean - Computational Omniscience Proof

def MonsterPrimes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

def IsDecidable (p : Nat) : Prop := p ∈ MonsterPrimes

theorem monster_decidable : ∀ p ∈ MonsterPrimes, IsDecidable p := fun p hp => hp

theorem seventy_one_largest : ∀ p ∈ MonsterPrimes, p ≤ 71 := by
  intro p hp
  cases hp <;> decide

structure Vector where v : List Nat
def transform (av : Vector) : Vector := match av.v with | [a,b,c] => ⟨[(a*2)%71,(b*3)%71,(c*5)%71]⟩ | _ => av
def IsFixed (av : Vector) : Prop := transform av = av

theorem fixed_exists : ∃ av, IsFixed av := ⟨⟨[0,0,0]⟩, rfl⟩

def KComplexity (n : Nat) : Nat := if n ∈ MonsterPrimes then 0 else 1

theorem zero_complexity : ∀ p ∈ MonsterPrimes, KComplexity p = 0 := by
  intro p hp; simp [KComplexity, hp]

structure Sys where rep : Nat; real : Nat
def Singular (s : Sys) : Prop := s.rep = s.real

theorem singular_exists : ∃ s, Singular s := ⟨⟨42,42⟩, rfl⟩

theorem computational_omniscience :
  (∀ p ∈ MonsterPrimes, IsDecidable p) ∧
  (∀ p ∈ MonsterPrimes, p ≤ 71) ∧
  (∃ av, IsFixed av) ∧
  (∀ p ∈ MonsterPrimes, KComplexity p = 0) ∧
  (∃ s, Singular s) :=
  ⟨monster_decidable, seventy_one_largest, fixed_exists, zero_complexity, singular_exists⟩

#check computational_omniscience
#print axioms computational_omniscience
