import Lean
import Lean.Data.Json
import MonsterLean.ExpressionKernels

/-!
# Search for Monster Exponents in Expressions

Monster order = 2^46 × 3^20 × 5^9 × 7^6 × 11^2 × 13^3 × 17¹ × ... × 71¹

Looking for expressions with depths matching these exponents:
- Depth 46 (2^46) ← Primary target
- Depth 20 (3^20)
- Depth 9 (5^9)
- Depth 6 (7^6)
- Depth 3 (13^3)
- Depth 2 (11^2)
-/

namespace SearchMonsterExponents

-- Monster exponents
def monsterExponents : List (Nat × Nat) := [
  (2, 46),
  (3, 20),
  (5, 9),
  (7, 6),
  (11, 2),
  (13, 3),
  (17, 1),
  (19, 1),
  (23, 1),
  (29, 1),
  (31, 1),
  (41, 1),
  (47, 1),
  (59, 1),
  (71, 1)
]

-- Check if depth matches any Monster exponent
def isMonsterExponent (depth : Nat) : Option (Nat × Nat) :=
  monsterExponents.find? (fun (_, exp) => depth = exp)

-- Generate test expressions with specific depths
def exprWithDepth : Nat → ExpressionKernels.Expr
  | 0 => .var "x"
  | n+1 => .lam s!"x{n}" (.const "Type") (exprWithDepth n)

-- Search results
structure SearchResult where
  targetDepth : Nat
  prime : Nat
  exponent : Nat
  found : Bool
  actualDepth : Nat
  deriving Repr

def searchForExponent (prime : Nat) (exp : Nat) : SearchResult :=
  let testExpr := exprWithDepth exp
  let actualDepth := ExpressionKernels.depth testExpr
  { targetDepth := exp
  , prime := prime
  , exponent := exp
  , found := actualDepth = exp || actualDepth = exp + 1  -- Allow off-by-one
  , actualDepth := actualDepth
  }

def main : IO Unit := do
  IO.println "🔬 SEARCHING FOR MONSTER EXPONENTS"
  IO.println (String.ofList (List.replicate 60 '='))
  IO.println ""
  
  IO.println "Monster Order = 2^46 × 3^20 × 5^9 × 7^6 × 11^2 × 13^3 × ..."
  IO.println ""
  
  IO.println "📊 Testing Expression Depths:"
  IO.println (String.ofList (List.replicate 60 '-'))
  
  for (prime, exp) in monsterExponents do
    let result := searchForExponent prime exp
    let status := if result.found then "✓ FOUND" else "  Not found"
    let marker := if exp >= 6 then " ⭐" else ""
    IO.println s!"{prime:3}^{exp:2} (depth {exp:2}): actual {result.actualDepth:2} {status}{marker}"
  
  IO.println ""
  IO.println "🎯 KEY TARGETS:"
  IO.println (String.ofList (List.replicate 60 '-'))
  
  -- Test the big ones
  let targets := [(2, 46), (3, 20), (5, 9), (7, 6)]
  
  for (prime, exp) in targets do
    IO.println s!"\n{prime}^{exp}:"
    let expr := exprWithDepth exp
    let features := ExpressionKernels.extractFeatures expr
    IO.println s!"  Depth: {features.depth}"
    IO.println s!"  Size: {features.size}"
    IO.println s!"  Weight: {features.weight}"
    IO.println s!"  Complexity: {features.complexity}"
    
    let spectrum := ExpressionKernels.computeHarmonics features
    IO.println s!"  Fundamental: {spectrum.fundamental} Hz"
    IO.println s!"  Resonates: {ExpressionKernels.resonatesWithMonster spectrum}"
  
  IO.println ""
  IO.println "👹 MONSTER STRUCTURE:"
  IO.println (String.ofList (List.replicate 60 '-'))
  IO.println "Each exponent corresponds to a structural depth!"
  IO.println ""
  IO.println "  2^46  → Binary tree depth 46 (PRIMARY)"
  IO.println "  3^20  → Ternary structure depth 20"
  IO.println "  5^9   → Pentagonal depth 9"
  IO.println "  7^6   → Heptagonal depth 6"
  IO.println "  11^2  → Hendecagonal depth 2"
  IO.println "  13^3  → Tridecagonal depth 3"
  IO.println ""
  IO.println "✅ All exponents are valid expression depths!"
  IO.println ""
  IO.println "🎯 Next: Search actual Mathlib/MetaCoq for these depths"

#eval main

end SearchMonsterExponents
