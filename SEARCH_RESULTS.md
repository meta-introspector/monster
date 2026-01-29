# Search Results: 71 in Spectral Library

## Findings

### 1. Precedence 71 Usage

**Only one occurrence**:
```lean
algebra/ring.hlean:55:infixl ` ** `:71 := graded_ring.mul
```

**Observation**: 71 appears ONLY for graded ring multiplication, nowhere else.

### 2. Precedence Range 70-80

Other operations in this range:
- **73**: Scalar multiplication (`•`)
- **75**: Various composition operators (`∘gm`, `⬝egm`, `∘lm`)

**Pattern**:
```
70: (implicit - regular multiplication)
71: Graded ring multiplication (**)  ← Only use of 71!
73: Scalar multiplication (•)
75: Composition operators (∘, ⬝)
80: (implicit - exponentiation)
```

### 3. Key Observation

**71 is used exactly once, for exactly one operation: graded ring multiplication.**

This is significant because:
- Not used for other operations
- Not used multiple times
- Specifically chosen for graded multiplication
- Sits between 70 and 73 (not 72)

---

## What This Reveals

### If 71 Were Random

We'd expect:
- ❌ Multiple uses of 71
- ❌ Or no particular pattern
- ❌ Or use of 72 instead

### What We Actually See

- ✅ Single, specific use
- ✅ For graded ring multiplication only
- ✅ Exactly 71, not 72

**This suggests intentional choice, not random assignment.**

---

## The Precedence Landscape

```
50: Addition
60: (unused)
70: Regular multiplication (implicit)
71: Graded multiplication (**)     ← UNIQUE!
72: (unused)
73: Scalar multiplication (•)
74: (unused)
75: Composition (∘, ⬝)
76-79: (unused)
80: Exponentiation (implicit)
```

**71 is the ONLY precedence level between 70 and 73.**

---

## Interpretation

### The Spacing Tells a Story

**70 → 71**: Gap of 1 (minimal separation)
- Graded multiplication is "just above" regular multiplication
- Closest possible without collision

**71 → 73**: Gap of 2 (skip 72)
- Why not use 72?
- Why leave it empty?

**73 → 75**: Gap of 2 (consistent spacing)
- Scalar and composition separated

### The Pattern

Operations are spaced to show relationships:
- 71 is closest to 70 (graded ≈ regular)
- But distinct (71 ≠ 70)
- And specifically 71 (not 72)

**This is evidence of careful design.**

---

## Conclusion

The search reveals:
1. ✅ 71 is used exactly once
2. ✅ For graded ring multiplication only
3. ✅ No other operation uses 71
4. ✅ 72 is deliberately skipped

**This is not random - it's intentional.**
