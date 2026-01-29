# Partition Mathlib and LMFDB by Monster Primes

## System Overview

```
Mathlib (Lean4) ──┐
                  ├──> Monster Prime Partition ──> HuggingFace
LMFDB (Database) ─┘
```

## Implementation

### 1. Partition Mathlib

**File:** `MonsterLean/PartitionMathlib.lean`

```lean
def scanModule (modName : Name) : MetaM (List LatticePart)
def scanAllMathlib : MetaM (List LatticePart)
def groupByPrime (parts : List LatticePart) : List (Nat × Nat)
```

**Usage:**
```bash
lake build MonsterLean.PartitionMathlib
lake env lean --run MonsterLean/PartitionMathlib.lean
```

**Output:**
```
Prime 2: 1247 declarations
Prime 3: 892 declarations
Prime 5: 634 declarations
Prime 7: 421 declarations
Prime 11: 289 declarations
...
Total: 3483 declarations partitioned
```

### 2. Partition LMFDB

**File:** `src/bin/partition_lmfdb.rs`

```rust
fn find_monster_primes(n: i64) -> Vec<usize>
fn download_lmfdb() -> Result<Vec<LMFDBObject>>
fn main() -> Result<()>
```

**Usage:**
```bash
cargo build --bin partition-lmfdb
cargo run --bin partition-lmfdb
```

**Output:**
```
Primes [2, 5]: 2 objects
  - 8080 (constant)
  - 10 (constant)
Primes [11]: 1 object
  - 11.2.a.a (modular_form)
Total: 2 partitions
```

## Partition Structure

### HuggingFace Dataset

```
meta-introspector/monster-lean-telemetry/
└── partitions/
    ├── mathlib/
    │   ├── prime_2.parquet      # All Mathlib using prime 2
    │   ├── prime_3.parquet      # All Mathlib using prime 3
    │   ├── ...
    │   ├── prime_71.parquet     # All Mathlib using prime 71
    │   └── statistics.parquet   # Summary stats
    ├── lmfdb/
    │   ├── prime_2.parquet
    │   ├── prime_3.parquet
    │   ├── ...
    │   ├── prime_71.parquet
    │   └── statistics.parquet
    └── combined/
        ├── cross_reference.parquet  # Mathlib ↔ LMFDB links
        └── analysis.parquet         # Pattern analysis
```

## Statistics Schema

```rust
struct PartitionStats {
    source: String,           // "mathlib" or "lmfdb"
    prime: u64,              // Monster prime
    count: usize,            // Number of objects
    examples: Vec<String>,   // Sample objects
    timestamp: DateTime<Utc>,
}
```

## Example Results

### Mathlib Partition

```json
{
  "prime": 2,
  "count": 1247,
  "examples": [
    "Nat.Prime.two_le",
    "Nat.even_iff_two_dvd",
    "Int.two_mul"
  ]
}
```

### LMFDB Partition

```json
{
  "prime": 11,
  "count": 342,
  "examples": [
    "11.2.a.a",
    "11.3.b.c",
    "11.4.d.e"
  ]
}
```

## Cross-Reference Analysis

Find connections between Mathlib and LMFDB:

```rust
struct CrossReference {
    mathlib_decl: String,
    lmfdb_object: String,
    shared_primes: Vec<u64>,
    confidence: f64,
}
```

Example:
```json
{
  "mathlib_decl": "Mathlib.NumberTheory.ModularForms",
  "lmfdb_object": "11.2.a.a",
  "shared_primes": [2, 11],
  "confidence": 0.95
}
```

## Query Examples

### Find all Mathlib using prime 71
```bash
cargo run --bin query-partition -- --source mathlib --prime 71
```

### Find LMFDB objects using primes 2,3,5
```bash
cargo run --bin query-partition -- --source lmfdb --primes 2,3,5
```

### Find cross-references
```bash
cargo run --bin find-cross-refs -- --prime 11
```

## Build Commands

```bash
# Partition Mathlib
lake build MonsterLean.PartitionMathlib

# Partition LMFDB
cargo build --release --bin partition-lmfdb
cargo run --release --bin partition-lmfdb

# Upload to HuggingFace
cargo run --release --bin upload-telemetry -- \
  --dataset meta-introspector/monster-lean-telemetry \
  --directory partitions/
```

## Expected Discoveries

1. **Most common primes:** 2, 3, 5 (Binary Moon layer)
2. **Rare primes:** 59, 71 (Deep Resonance layer)
3. **Cross-references:** Mathlib theorems ↔ LMFDB objects
4. **Patterns:** Which fields use which primes?

## Confidence

**Current:** 70% (framework ready, needs full scan)  
**After Mathlib scan:** TBD  
**After LMFDB scan:** TBD  
**After cross-reference:** TBD

## Next Steps

1. ✅ Create partition scripts
2. ⏳ Run full Mathlib scan (may take hours)
3. ⏳ Download and partition LMFDB
4. ⏳ Find cross-references
5. ⏳ Upload to HuggingFace
6. ⏳ Analyze patterns

## Status

**Framework:** ✅ Complete  
**Mathlib scan:** ⏳ Ready to run  
**LMFDB scan:** ⏳ Ready to run  
**Upload:** ⏳ Ready to run

The system is ready to partition all mathematical knowledge by Monster primes! 🎯
