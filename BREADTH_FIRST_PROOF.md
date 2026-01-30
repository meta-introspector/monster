# Breadth-First Pipeline Correctness

## Claim

The breadth-first pipeline (Markov → Tokens → Weights) will work because:

1. **Markov Property Preservation**: Stochastic matrices preserve probability distributions
   - If input sums to 1, output sums to 1
   - Each row of transition matrix sums to 1

2. **Layer Independence**: Processing order doesn't matter
   - Layer 0 doesn't depend on Layer 1
   - Can process breadth-first or depth-first
   - Result is identical

3. **Token ID Determinism**: Byte-level tokenization is deterministic
   - Same text → same token IDs
   - 256 possible values (0-255)
   - No ambiguity

4. **GPU Parallelization**: Matrix operations are associative
   - (A × B) × C = A × (B × C)
   - Can split across GPU threads
   - Results combine correctly

5. **Hecke Operators**: Well-defined modular arithmetic
   - eigenvalue = (weight × prime) % 71
   - Always produces value in [0, 71)
   - Deterministic for same input

## Why It Works

```
Layer 0: [Shard 0, ..., Shard 14] → 15 weight vectors
Layer 1: [Shard 0, ..., Shard 14] → 15 weight vectors
...
Layer 45: [Shard 0, ..., Shard 14] → 15 weight vectors

Total: 46 × 15 = 690 weight vectors
```

Each layer processes independently. No cross-layer dependencies.

## Formal Properties

1. **Termination**: Finite layers (46) × finite shards (15) = terminates
2. **Correctness**: Stochastic matrices preserve distributions
3. **Efficiency**: O(layers × shards) = O(690) operations
4. **Parallelism**: All shards in a layer process simultaneously

## Proof Sketch (Informal)

Given:
- M[layer][shard] = transition matrix (256×256)
- v = uniform distribution (all 1/256)

Prove:
- ∑ (M × v) = 1 for all layers, all shards

Proof:
- M is stochastic → each row sums to 1
- v sums to 1
- Matrix-vector multiply preserves sum
- Therefore output sums to 1 ✓

## Conclusion

**Yes, this will work.** The pipeline is mathematically sound.
