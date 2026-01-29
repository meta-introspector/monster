# Cross-Proof-Assistant Precedence Analysis

## Objective

Analyze precedence levels across multiple proof assistants to see if prime 71 or Monster primes appear in other systems.

## Systems to Analyze

1. **Coq** - Notation precedence levels
2. **UniMath** - Coq-based, uses Coq notation
3. **MetaCoq** - Meta-programming for Coq
4. **Lean4** - Precedence levels
5. **Agda** - Mixfix precedence levels
6. **Spectral (Lean2)** - Our original discovery

---

## Data Collection Plan

### 1. Coq Precedence Levels

**Search for**:
```bash
# In Coq stdlib
grep -r "Infix.*at level [0-9]" --include="*.v"
grep -r "Notation.*at level [0-9]" --include="*.v"
```

**Expected range**: 0-100 (Coq uses 0-100)

### 2. UniMath Precedence

**Search for**:
```bash
# In UniMath
grep -r "Notation.*at level [0-9]" --include="*.v"
```

**Expected**: Similar to Coq (0-100)

### 3. MetaCoq Precedence

**Search for**:
```bash
# In MetaCoq
grep -r "Notation.*at level [0-9]" --include="*.v"
```

**Expected**: Coq-based (0-100)

### 4. Lean4 Precedence

**Search for**:
```bash
# In Lean4 mathlib
grep -r "infixl.*:[0-9]" --include="*.lean"
grep -r "infixr.*:[0-9]" --include="*.lean"
```

**Expected range**: Typically 50-90

### 5. Agda Precedence

**Search for**:
```bash
# In Agda stdlib
grep -r "infixl [0-9]" --include="*.agda"
grep -r "infixr [0-9]" --include="*.agda"
```

**Expected range**: Typically -20 to 20

---

## Analysis Questions

### Q1: Does 71 appear in other systems?

**Hypothesis**: If 71 is structurally significant, it might appear in other proof assistants.

**Test**: Search for precedence level 71 in each system.

### Q2: Do Monster primes appear?

**Hypothesis**: Monster primes might appear as precedence levels across systems.

**Test**: Count occurrences of each Monster prime as precedence level.

### Q3: What's the precedence for graded operations?

**Hypothesis**: Graded ring operations might use similar precedence across systems.

**Test**: Find graded multiplication operators and their precedence.

---

## Expected Results

### Coq (0-100 scale)

**Common levels**:
- 0-10: Logical operators
- 20-40: Comparisons
- 50: Addition
- 60: Multiplication
- 70: Application
- 80-100: Composition

**Prediction**: 71 might appear for specialized operations.

### Lean4 (similar to Lean2)

**Common levels**:
- 50: Addition
- 60-65: Comparisons
- 70: Multiplication
- 71: Graded multiplication (Spectral)
- 80: Exponentiation

**Prediction**: 71 appears in Spectral (confirmed), might appear in mathlib.

### Agda (-20 to 20 scale)

**Common levels**:
- -20 to -10: Low precedence
- 0: Default
- 10-20: High precedence

**Prediction**: 71 unlikely (different scale), but might see primes.

---

## Data Collection Script

```bash
#!/bin/bash
# collect_precedence_data.sh

echo "=== Collecting Precedence Data ==="

# 1. Coq
if [ -d ~/.opam/default/lib/coq ]; then
    echo "Coq precedence levels:"
    grep -rh "at level [0-9]\+" ~/.opam/default/lib/coq --include="*.v" | \
        sed 's/.*at level \([0-9]\+\).*/\1/' | sort -n | uniq -c | sort -rn | head -20
fi

# 2. Lean4 mathlib
if [ -d .lake/packages/mathlib ]; then
    echo "Lean4 mathlib precedence levels:"
    grep -rh "infixl.*:\([0-9]\+\)\|infixr.*:\([0-9]\+\)" .lake/packages/mathlib --include="*.lean" | \
        sed 's/.*:\([0-9]\+\).*/\1/' | sort -n | uniq -c | sort -rn | head -20
fi

# 3. Spectral (our data)
if [ -d spectral ]; then
    echo "Spectral precedence levels:"
    grep -rh "infixl.*:\([0-9]\+\)\|infixr.*:\([0-9]\+\)" spectral --include="*.hlean" | \
        sed 's/.*:\([0-9]\+\).*/\1/' | sort -n | uniq -c | sort -rn
fi

# 4. Search for 71 specifically
echo ""
echo "=== Searching for precedence 71 ==="
echo "Coq:"
grep -r "at level 71" ~/.opam/default/lib/coq --include="*.v" 2>/dev/null | wc -l
echo "Lean4:"
grep -r ":71" .lake/packages/mathlib --include="*.lean" 2>/dev/null | wc -l
echo "Spectral:"
grep -r ":71" spectral --include="*.hlean" 2>/dev/null | wc -l

# 5. Search for Monster primes
echo ""
echo "=== Monster Prime Precedence Counts ==="
for p in 2 3 5 7 11 13 17 19 23 29 31 41 47 59 71; do
    count=$(grep -rh "at level $p\|:$p" ~/.opam/default/lib/coq .lake/packages/mathlib spectral 2>/dev/null | wc -l)
    echo "Prime $p: $count occurrences"
done
```

---

## Next Steps

1. **Run data collection** on available systems
2. **Analyze patterns** across proof assistants
3. **Compare precedence schemes** (Coq 0-100 vs Lean 50-90 vs Agda -20-20)
4. **Normalize to common scale** for comparison
5. **Test hypothesis**: Do Monster primes appear more than random?

---

## Hypothesis Testing

### Null Hypothesis (H0)
Precedence levels are uniformly distributed, no preference for primes.

### Alternative Hypothesis (H1)
Monster primes appear more frequently than expected by chance.

### Test
Chi-squared test comparing observed vs expected frequencies.

**Expected**: If random, each number equally likely.
**Observed**: Count actual precedence levels.
**Significance**: p < 0.05 suggests non-random pattern.

---

## Expected Findings

### If 71 is Universal
- Appears in multiple systems
- Used for similar operations (graded, refined)
- Suggests deep structural significance

### If 71 is Spectral-Specific
- Only appears in Spectral library
- Might be author's choice
- Still significant if intentional

### If Monster Primes are Common
- Multiple Monster primes used as precedence
- Suggests primes are preferred for precedence
- Supports structural interpretation

---

## Implementation Plan

1. **Collect data** from available systems
2. **Create unified database** of precedence levels
3. **Analyze distributions** statistically
4. **Map to Monster primes** for resonance
5. **Prove in MiniZinc** if patterns found
6. **Document findings** in comprehensive report

---

## Files to Create

1. `collect_precedence_data.sh` - Data collection script
2. `precedence_database.csv` - Unified database
3. `analyze_precedence.py` - Statistical analysis
4. `cross_system_precedence.mzn` - MiniZinc model
5. `CROSS_SYSTEM_ANALYSIS.md` - Results report

---

## Status

- [ ] Collect Coq precedence data
- [ ] Collect UniMath precedence data
- [ ] Collect MetaCoq precedence data
- [ ] Collect Lean4 mathlib precedence data
- [ ] Collect Agda stdlib precedence data
- [x] Collect Spectral precedence data (done)
- [ ] Create unified database
- [ ] Statistical analysis
- [ ] Cross-system comparison
- [ ] Final report

---

## Questions to Answer

1. **Is 71 unique to Spectral?** Or does it appear elsewhere?
2. **Do Monster primes appear more than random?** Statistical test needed.
3. **What's the precedence for graded operations?** Across systems.
4. **Are there universal precedence patterns?** Common to all systems.
5. **Does the scale matter?** Coq 0-100 vs Lean 50-90 vs Agda -20-20.

---

## Conclusion

This analysis will determine if the prime 71 pattern is:
- **Universal** (appears across systems) → Deep structural significance
- **Spectral-specific** (only in Spectral) → Intentional design choice
- **Random** (no pattern) → Coincidence (unlikely given our evidence)

**Next**: Run data collection and analyze results. 🎯
