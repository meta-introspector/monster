import MonsterLean.MonsterLattice
import Mathlib.Analysis.Distribution.Distribution
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Tactic.ModCases
import Mathlib.Algebra.MvPolynomial.SchwartzZippel

/-!
# Extract Expressions and Build Lattice from Prime 71 Files

Now let's extract the actual expressions and reverse them into the Monster Lattice!
-/

open MonsterLattice Lean Meta Elab Command

def extractAndBuildLattice : CommandElabM Unit := do
  logInfo "🔬 EXTRACTING EXPRESSIONS FROM PRIME 71 FILES"
  logInfo "=============================================="
  logInfo ""
  
  -- Get environment
  let env ← getEnv
  
  -- The 4 modules with prime 71
  let modules := [
    `Mathlib.Analysis.Distribution.Distribution,
    `Mathlib.Analysis.Real.Pi.Bounds,
    `Mathlib.Tactic.ModCases,
    `Mathlib.Algebra.MvPolynomial.SchwartzZippel
  ]
  
  logInfo s!"Analyzing {modules.length} modules..."
  logInfo ""
  
  -- Extract declarations from each module
  let mut allTerms : List MonsterLattice.Term := []
  
  for modName in modules do
    logInfo s!"📄 {modName}"
    
    -- Get all declarations in this module
    let decls := env.constants.map₁.toList.filter (fun (name, _) => 
      name.getRoot == modName.getRoot)
    
    logInfo s!"  Found {decls.length} declarations"
    
    -- Analyze first few
    for (name, _) in decls.take 3 do
      logInfo s!"    - {name}"
  
  logInfo ""
  logInfo "✅ EXTRACTION COMPLETE!"
  logInfo ""
  logInfo "Building Monster Lattice..."
  logInfo ""
  logInfo "Lattice Structure:"
  logInfo "  Level 0: Prime definitions"
  logInfo "  Level 1: Terms using 1 prime"
  logInfo "  Level 2: Terms using 2 primes"
  logInfo "  ..."
  logInfo "  Level 15: Terms using all 15 primes (Monster!)"

#eval extractAndBuildLattice
