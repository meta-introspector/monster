# Bisimulation Proof: Multi-Language Precedence Survey

## Objective

Rewrite the precedence survey in 5 languages and prove they are bisimilar:
1. **Rust** - Performance implementation
2. **Lean4** - Formal specification
3. **MiniZinc** - Constraint model
4. **Prolog** - Logic programming
5. **Python** (reference) - Original implementation

Then prove: All implementations produce identical results (bisimulation).

---

## The Core Algorithm

### Input
- List of directories to scan
- File patterns to match
- Precedence extraction patterns

### Process
1. Walk directory tree
2. Find matching files
3. Extract precedence declarations
4. Record metadata (file, line, operator, precedence)
5. Output structured data

### Output
- List of precedence records
- Each record: (system, project, file, line, operator, precedence, git_url, commit, branch)

---

## Implementation Plan

### 1. Rust Implementation (Performance)

**File**: `src/bin/precedence_survey.rs`

**Status**: ✓ Already written

**Features**:
- Fast file walking
- Regex parsing
- Parquet output
- Git metadata extraction

### 2. Lean4 Implementation (Specification)

**File**: `MonsterLean/PrecedenceSurvey.lean`

**Purpose**: Formal specification of the algorithm

**Features**:
- Pure functional
- Proven correct
- Type-safe
- Reference implementation

### 3. MiniZinc Implementation (Constraints)

**File**: `minizinc/precedence_survey.mzn`

**Purpose**: Constraint-based model

**Features**:
- Declarative
- Constraint satisfaction
- Verifiable
- Solver-independent

### 4. Prolog Implementation (Logic)

**File**: `prolog/precedence_survey.pl`

**Purpose**: Logic programming approach

**Features**:
- Declarative rules
- Pattern matching
- Backtracking search
- Logic-based

### 5. Python Implementation (Reference)

**File**: `quarantine/scripts/precedence_survey.py`

**Purpose**: Reference implementation (for comparison only)

**Status**: Quarantined, not tracked

---

## Bisimulation Proof Strategy

### Definition

Two implementations are **bisimilar** if:
1. Given same input, produce same output
2. Maintain same state transitions
3. Preserve behavioral equivalence

### Proof Method

**For each pair of implementations (A, B)**:

1. **Input equivalence**: Same test cases
2. **Output equivalence**: Compare results
3. **State equivalence**: Compare intermediate states
4. **Behavioral equivalence**: Same observable behavior

### Test Cases

**Test 1: Empty directory**
- Input: Empty directory
- Expected: Empty result set

**Test 2: Single file, single precedence**
- Input: One file with one precedence declaration
- Expected: One record

**Test 3: Multiple files, multiple precedences**
- Input: Directory with multiple files
- Expected: All precedences found

**Test 4: Spectral library**
- Input: spectral/ directory
- Expected: Known precedence declarations (71, 73, 75, etc.)

**Test 5: Lean4 mathlib**
- Input: .lake/packages/mathlib
- Expected: Known precedence declarations

---

## Implementation 1: Rust (Already Done)

**File**: `src/bin/precedence_survey.rs`

**Status**: ✓ Complete

**Key functions**:
- `scan_spectral()` - Scan Spectral library
- `scan_lean4()` - Scan Lean4 mathlib
- `scan_coq()` - Scan Coq stdlib
- `parse_lean_precedence()` - Extract Lean precedence
- `parse_coq_precedence()` - Extract Coq precedence

---

## Implementation 2: Lean4 (Specification)

**File**: `MonsterLean/PrecedenceSurvey.lean`

```lean
import Std.Data.List.Basic
import Std.Data.String.Basic

namespace PrecedenceSurvey

/-- A precedence record -/
structure PrecedenceRecord where
  system : String
  project : String
  file : String
  line : Nat
  operator : String
  precedence : Nat
  git_url : String
  commit : String
  branch : String

/-- Extract precedence from Lean syntax -/
def parseLeanPrecedence (line : String) : Option (String × Nat) :=
  -- Parse: infixl ` ** `:71 := graded_ring.mul
  sorry -- Implementation

/-- Extract precedence from Coq syntax -/
def parseCoqPrecedence (line : String) : Option (String × Nat) :=
  -- Parse: Notation "x + y" := (add x y) (at level 50).
  sorry -- Implementation

/-- Scan a file for precedence declarations -/
def scanFile (path : String) (parser : String → Option (String × Nat)) : 
  IO (List PrecedenceRecord) :=
  sorry -- Implementation

/-- Scan a directory recursively -/
def scanDirectory (dir : String) (pattern : String) 
  (parser : String → Option (String × Nat)) : 
  IO (List PrecedenceRecord) :=
  sorry -- Implementation

/-- Main survey function -/
def survey : IO (List PrecedenceRecord) := do
  let spectral ← scanDirectory "spectral" "*.hlean" parseLeanPrecedence
  let lean4 ← scanDirectory ".lake/packages/mathlib" "*.lean" parseLeanPrecedence
  return spectral ++ lean4

end PrecedenceSurvey
```

---

## Implementation 3: MiniZinc (Constraints)

**File**: `minizinc/precedence_survey.mzn`

```minizinc
% Precedence Survey as Constraint Satisfaction Problem

% Input: Known precedence declarations
array[1..N] of string: files;
array[1..N] of int: lines;
array[1..N] of string: operators;
array[1..N] of int: precedences;

% Constraints: Verify precedence properties

% 1. All precedences are positive
constraint forall(i in 1..N)(precedences[i] > 0);

% 2. Precedence 71 exists
constraint exists(i in 1..N)(precedences[i] = 71);

% 3. Precedence 71 is used for specific operators
constraint forall(i in 1..N)(
  precedences[i] = 71 -> (operators[i] = "**" \/ operators[i] = "mod_cases")
);

% 4. Count occurrences of each Monster prime
function var int: count_precedence(int: p) =
  sum(i in 1..N)(precedences[i] = p);

% Solve
solve satisfy;

% Output
output [
  "Precedence 71 count: ", show(count_precedence(71)), "\n",
  "All precedences valid: ", show(forall(i in 1..N)(precedences[i] > 0)), "\n"
];
```

---

## Implementation 4: Prolog (Logic)

**File**: `prolog/precedence_survey.pl`

```prolog
% Precedence Survey in Prolog

% Facts: Precedence declarations
precedence(lean2, spectral, 'spectral/algebra/ring.hlean', 55, '**', 71).
precedence(lean4, mathlib, 'Mathlib/Tactic/ModCases.lean', 186, 'mod_cases', 71).

% Rules: Query precedence declarations

% Find all occurrences of a specific precedence
find_precedence(P, System, Project, File, Line, Op) :-
    precedence(System, Project, File, Line, Op, P).

% Count occurrences of a precedence
count_precedence(P, Count) :-
    findall(1, precedence(_, _, _, _, _, P), List),
    length(List, Count).

% Find all Monster primes used as precedence
monster_prime(2).
monster_prime(3).
monster_prime(5).
monster_prime(7).
monster_prime(11).
monster_prime(13).
monster_prime(17).
monster_prime(19).
monster_prime(23).
monster_prime(29).
monster_prime(31).
monster_prime(41).
monster_prime(47).
monster_prime(59).
monster_prime(71).

monster_precedence(P) :-
    monster_prime(P),
    precedence(_, _, _, _, _, P).

% Query: Which Monster primes are used as precedence?
% ?- findall(P, monster_precedence(P), Primes).

% Query: How many times is 71 used?
% ?- count_precedence(71, Count).

% Query: What operators use precedence 71?
% ?- find_precedence(71, System, Project, File, Line, Op).
```

---

## Bisimulation Test Suite

**File**: `tests/bisimulation_test.rs`

```rust
#[cfg(test)]
mod bisimulation_tests {
    use super::*;

    #[test]
    fn test_rust_lean4_equivalence() {
        // Run Rust implementation
        let rust_results = run_rust_survey();
        
        // Run Lean4 implementation
        let lean4_results = run_lean4_survey();
        
        // Compare results
        assert_eq!(rust_results.len(), lean4_results.len());
        for (r, l) in rust_results.iter().zip(lean4_results.iter()) {
            assert_eq!(r.precedence, l.precedence);
            assert_eq!(r.operator, l.operator);
        }
    }

    #[test]
    fn test_rust_minizinc_equivalence() {
        // Run Rust implementation
        let rust_results = run_rust_survey();
        
        // Run MiniZinc model
        let minizinc_results = run_minizinc_survey();
        
        // Verify constraints are satisfied
        assert!(minizinc_results.all_valid);
        assert_eq!(rust_results.count_71(), minizinc_results.count_71);
    }

    #[test]
    fn test_rust_prolog_equivalence() {
        // Run Rust implementation
        let rust_results = run_rust_survey();
        
        // Query Prolog
        let prolog_results = query_prolog("count_precedence(71, Count)");
        
        // Compare counts
        assert_eq!(rust_results.count_71(), prolog_results.count);
    }
}
```

---

## Bisimulation Proof in Lean4

**File**: `MonsterLean/BisimulationProof.lean`

```lean
import MonsterLean.PrecedenceSurvey

namespace BisimulationProof

/-- Two implementations are bisimilar if they produce the same output -/
def bisimilar (impl1 impl2 : IO (List PrecedenceRecord)) : Prop :=
  ∀ input, impl1 input = impl2 input

/-- Theorem: Rust and Lean4 implementations are bisimilar -/
theorem rust_lean4_bisimilar : 
  bisimilar rust_survey lean4_survey := by
  sorry -- Proof by testing on all inputs

/-- Theorem: All implementations are pairwise bisimilar -/
theorem all_bisimilar :
  bisimilar rust_survey lean4_survey ∧
  bisimilar rust_survey minizinc_survey ∧
  bisimilar rust_survey prolog_survey := by
  sorry -- Proof by composition

end BisimulationProof
```

---

## Implementation Timeline

### Phase 1: Lean4 Specification (2 hours)
- [ ] Write formal specification
- [ ] Define data structures
- [ ] Implement core algorithm
- [ ] Test on Spectral

### Phase 2: MiniZinc Model (1 hour)
- [ ] Define constraint model
- [ ] Encode precedence rules
- [ ] Test on known data
- [ ] Verify constraints

### Phase 3: Prolog Implementation (1 hour)
- [ ] Define facts and rules
- [ ] Implement queries
- [ ] Test on known data
- [ ] Verify logic

### Phase 4: Bisimulation Tests (2 hours)
- [ ] Write test suite
- [ ] Run all implementations
- [ ] Compare outputs
- [ ] Document differences

### Phase 5: Formal Proof (3 hours)
- [ ] Prove pairwise bisimulation
- [ ] Prove transitivity
- [ ] Prove completeness
- [ ] Document proof

**Total**: ~9 hours

---

## Success Criteria

### Implementation Complete
- ✓ Rust implementation works
- ✓ Lean4 specification compiles
- ✓ MiniZinc model solves
- ✓ Prolog queries succeed

### Bisimulation Proven
- ✓ All implementations produce same output
- ✓ Test suite passes
- ✓ Formal proof in Lean4
- ✓ Documentation complete

### Evidence Complete
- ✓ Full citations for all data
- ✓ Git URLs, commits, branches
- ✓ Parquet files with metadata
- ✓ Reproducible builds

---

## Next Steps

1. **Implement Lean4 specification** - Formal algorithm
2. **Implement MiniZinc model** - Constraint verification
3. **Implement Prolog rules** - Logic queries
4. **Write test suite** - Bisimulation tests
5. **Prove equivalence** - Formal proof in Lean4
6. **Document everything** - Complete evidence

**Goal**: Prove all implementations are bisimilar, providing multiple independent verifications of our findings. 🎯
