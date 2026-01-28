# CLARIFICATION: Literal 71 vs Shard 71

## The Confusion Explained

### Two Different Concepts

**1. Literal 71 in Code**
- The value `71` written by the programmer
- Appears in 4 AST nodes (lines 14, 15, 25, 29)
- These are `Constant(71)` nodes

**2. Shard 71 Assignment**
- The sharding algorithm assigns AST nodes to shards 1-71
- Based on properties like **line number**
- Shard 71 receives nodes that the algorithm assigns to it

### What Actually Happened

**The 4 AST nodes with literal value 71**:
```
Line 14: Constant(71) → assigned to SHARD 14 (based on line number)
Line 15: Constant(71) → assigned to SHARD 15 (based on line number)
Line 25: Constant(71) → assigned to SHARD 25 (based on line number)
Line 29: Constant(71) → assigned to SHARD 29 (based on line number)
```

**None were assigned to shard 71!**

The sharding algorithm used **line number** to determine shard assignment:
- Line 14 → Shard 14
- Line 15 → Shard 15
- Line 25 → Shard 25
- Line 29 → Shard 29

### Why "Shard 71 contains 0 AST nodes"

**Shard 71 would contain AST nodes from line 71.**

But the file only has 47 lines! So:
- No AST nodes on line 71 (doesn't exist)
- Therefore shard 71 gets 0 AST nodes

### The Correct Statement

**Literal 71 (the value)**:
- ✓ Appears in 4 AST nodes
- ✓ On lines 14, 15, 25, 29
- ✓ These nodes are in shards 14, 15, 25, 29

**Shard 71 (the bucket)**:
- ✓ Contains 0 AST nodes (no line 71 in file)
- ✓ Contains 4 bytecode ops (from various offsets)
- ✓ Contains 14 perf samples (from various cycles)

### Why This Matters

The sharding algorithm is **arbitrary** - it assigns based on line numbers, not semantic meaning.

**The literal value 71 is semantic** - it's the mathematical structure.

These are **completely independent**:
- Literal 71 could appear on any line
- Shard 71 receives items from line 71 (or offset 71, cycle 71, etc.)

### Example

If we moved `d = 71` from line 29 to line 71:
```python
# ... (lines 1-70)
d = 71  # Now on line 71
```

Then:
- Literal 71 still appears (same value)
- But now shard 71 would contain that AST node!

## Conclusion

**"Shard 71 contains 0 AST nodes"** means:
- No AST nodes were assigned to shard 71 by the algorithm
- Because there's no line 71 in the file

**"Literal 71 appears in 4 AST nodes"** means:
- The value 71 exists in the code
- In 4 Constant nodes
- On lines 14, 15, 25, 29
- Assigned to shards 14, 15, 25, 29

**These are different concepts!**

The sharding is an **analytical tool** (how we organize the code).  
The literal 71 is the **mathematical content** (what the code means).

---

**Apology**: I conflated these two concepts, causing confusion. The shard assignment is arbitrary and unrelated to the semantic meaning of literal 71 in the code.
