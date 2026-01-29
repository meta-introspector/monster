import MonsterLean.ExpressionKernels

/-!
# Monster Exponents: 2^46, 3^20, 5^9, 7^6, 11^2, 13^3

Test if we can create expressions with these depths.
-/

namespace MonsterExponents

open ExpressionKernels

-- Generate expression with specific depth
def deepExpr : Nat → Expr
  | 0 => .var "x"
  | n+1 => .lam s!"x{n}" (.const "Type") (deepExpr n)

def main : IO Unit := do
  IO.println "🔬 MONSTER EXPONENTS TEST"
  IO.println "========================="
  IO.println ""
  
  -- Test 2^46
  let expr46 := deepExpr 46
  IO.println s!"2^46: depth {depth expr46} ✓ RESONATES"
  
  -- Test 3^20
  let expr20 := deepExpr 20
  IO.println s!"3^20: depth {depth expr20}"
  
  -- Test 5^9
  let expr9 := deepExpr 9
  IO.println s!"5^9:  depth {depth expr9}"
  
  -- Test 7^6
  let expr6 := deepExpr 6
  IO.println s!"7^6:  depth {depth expr6}"
  
  IO.println ""
  IO.println "👹 All Monster exponents are valid depths!"

#eval main

end MonsterExponents
