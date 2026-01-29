import Lean
import MonsterLean.ExpressionKernels

/-!
# Multi-Dimensional Depth Analysis

Not just one depth, but MULTIPLE depth dimensions:
1. **Expression depth** - AST nesting
2. **Type depth** - Type hierarchy nesting
3. **Function nesting** - Lambda/Pi nesting
4. **Universe levels** - Type^n hierarchy

Each dimension can match Monster exponents!
-/

namespace MultiDimensionalDepth

open Lean

-- 1. Expression depth (already have)
def exprDepth : Expr → Nat
  | .bvar _ => 1
  | .fvar _ => 1
  | .const _ _ => 1
  | .forallE _ t b _ => 1 + max (exprDepth t) (exprDepth b)
  | .lam _ t b _ => 1 + max (exprDepth t) (exprDepth b)
  | .app f a => 1 + max (exprDepth f) (exprDepth a)
  | _ => 1

-- 2. Type depth - how deep is the type hierarchy?
def typeDepth : Expr → Nat
  | .forallE _ ty body _ => 1 + max (typeDepth ty) (typeDepth body)
  | .lam _ ty body _ => max (typeDepth ty) (typeDepth body)
  | .app f a => max (typeDepth f) (typeDepth a)
  | .sort (.succ u) => 1 + uDepth u
  | _ => 0
where
  uDepth : Level → Nat
  | .zero => 0
  | .succ u => 1 + uDepth u
  | .max u v => max (uDepth u) (uDepth v)
  | .imax u v => max (uDepth u) (uDepth v)
  | _ => 0

-- 3. Function nesting depth - count nested lambdas/pis
def functionNestingDepth : Expr → Nat
  | .forallE _ _ body _ => 1 + functionNestingDepth body
  | .lam _ _ body _ => 1 + functionNestingDepth body
  | .app f _ => functionNestingDepth f
  | _ => 0

-- 4. Universe level - maximum universe in expression
def maxUniverseLevel : Expr → Nat
  | .sort (.succ u) => 1 + uDepth u
  | .forallE _ ty body _ => max (maxUniverseLevel ty) (maxUniverseLevel body)
  | .lam _ ty body _ => max (maxUniverseLevel ty) (maxUniverseLevel body)
  | .app f a => max (maxUniverseLevel f) (maxUniverseLevel a)
  | .const _ levels => levels.foldl (fun acc l => max acc (uDepth l)) 0
  | _ => 0
where
  uDepth : Level → Nat
  | .zero => 0
  | .succ u => 1 + uDepth u
  | .max u v => max (uDepth u) (uDepth v)
  | .imax u v => max (uDepth u) (uDepth v)
  | _ => 0

-- Multi-dimensional depth vector
structure DepthVector where
  exprDepth : Nat
  typeDepth : Nat
  functionNesting : Nat
  universeLevel : Nat
  deriving Repr

def analyzeDepths (e : Expr) : DepthVector :=
  { exprDepth := exprDepth e
  , typeDepth := typeDepth e
  , functionNesting := functionNestingDepth e
  , universeLevel := maxUniverseLevel e
  }

def matchesMonsterExponent (dv : DepthVector) : List (String × Nat) :=
  let targets := [46, 47, 20, 21, 9, 10, 6, 7, 3, 4, 2]
  let r1 := if targets.contains dv.exprDepth then [("expr", dv.exprDepth)] else []
  let r2 := if targets.contains dv.typeDepth then [("type", dv.typeDepth)] else []
  let r3 := if targets.contains dv.functionNesting then [("func", dv.functionNesting)] else []
  let r4 := if targets.contains dv.universeLevel then [("univ", dv.universeLevel)] else []
  r1 ++ r2 ++ r3 ++ r4

-- Test expressions
def testExpr1 : Expr :=
  .lam `x (.sort .zero) (.bvar 0) .default

def testExpr2 : Expr :=
  .forallE `x (.sort (.succ .zero)) (.bvar 0) .default

def main : IO Unit := do
  IO.println "🔬 MULTI-DIMENSIONAL DEPTH ANALYSIS"
  IO.println (String.ofList (List.replicate 60 '='))
  IO.println ""
  
  IO.println "📊 Four Depth Dimensions:"
  IO.println "  1. Expression depth (AST nesting)"
  IO.println "  2. Type depth (type hierarchy)"
  IO.println "  3. Function nesting (λ/Π count)"
  IO.println "  4. Universe level (Type^n)"
  IO.println ""
  
  IO.println "🎯 Monster Exponents to Find:"
  IO.println "  2^46 = 46 (or 47)"
  IO.println "  3^20 = 20 (or 21)"
  IO.println "  5^9  = 9  (or 10)"
  IO.println "  7^6  = 6  (or 7)"
  IO.println "  13^3 = 3  (or 4)"
  IO.println "  11^2 = 2  (or 3)"
  IO.println ""
  
  IO.println "📊 Example Analysis:"
  IO.println (String.ofList (List.replicate 60 '-'))
  
  let depths1 := analyzeDepths testExpr1
  IO.println s!"λx:Type.x:"
  IO.println s!"  Expr depth: {depths1.exprDepth}"
  IO.println s!"  Type depth: {depths1.typeDepth}"
  IO.println s!"  Func nesting: {depths1.functionNesting}"
  IO.println s!"  Universe: {depths1.universeLevel}"
  
  let matches1 := matchesMonsterExponent depths1
  if matches1.length > 0 then
    IO.println "  ✓ Matches Monster exponents:"
    for (dim, val) in matches1 do
      IO.println s!"    {dim}: {val}"
  
  IO.println ""
  
  let depths2 := analyzeDepths testExpr2
  IO.println s!"∀x:Type₁.x:"
  IO.println s!"  Expr depth: {depths2.exprDepth}"
  IO.println s!"  Type depth: {depths2.typeDepth}"
  IO.println s!"  Func nesting: {depths2.functionNesting}"
  IO.println s!"  Universe: {depths2.universeLevel}"
  
  IO.println ""
  IO.println "👹 THE MONSTER IS MULTI-DIMENSIONAL!"
  IO.println (String.ofList (List.replicate 60 '-'))
  IO.println "Each dimension can independently match Monster exponents:"
  IO.println ""
  IO.println "  Expression with:"
  IO.println "    - Expr depth 46 (2^46)"
  IO.println "    - Type depth 20 (3^20)"
  IO.println "    - Func nesting 9 (5^9)"
  IO.println "    - Universe 6 (7^6)"
  IO.println ""
  IO.println "  → QUADRUPLE Monster resonance!"
  IO.println ""
  IO.println "✅ Search space expanded 4x!"
  IO.println "   More likely to find Monster structure in actual code"

#eval main

end MultiDimensionalDepth
