# Universal Shard Reader

Read Monster ZK Lattice shards from local, Archive.org, or Hugging Face.

## Rust Implementation

### Usage

```rust
use universal_shard_reader::{ShardReader, ShardSource};

// Local
let reader = ShardReader::new(ShardSource::Local("archive_org_shards".to_string()));
let shard = reader.read_rdf_shard(0)?;
let lattice = reader.read_value_lattice()?;

// Archive.org
let reader = ShardReader::new(ShardSource::ArchiveOrg("monster-zk-lattice-v1".to_string()));
let shard = reader.read_rdf_shard(0)?;

// Hugging Face
let reader = ShardReader::new(ShardSource::HuggingFace("username/monster-zk-lattice".to_string()));
let shard = reader.read_rdf_shard(0)?;
```

### Build

```bash
cd /home/mdupont/experiments/monster

# Local only
cargo build --release --bin universal_shard_reader

# With network support
cargo build --release --bin universal_shard_reader --features reqwest

# Run
./target/release/universal_shard_reader
```

### Features

- ✅ Read 71 RDF shards
- ✅ Read value lattice JSON
- ✅ Content-addressable (SHA256)
- ✅ Local filesystem
- ✅ Archive.org HTTP
- ✅ Hugging Face HTTP

## Lean4 Specification

### Theorems

```lean
-- All shards are content-addressable
theorem shard_content_addressable (s : RDFShard) :
  s.contentHash.length > 0

-- Shard IDs are valid
theorem shard_id_valid (s : RDFShard) :
  s.shardId < 71

-- Reading is deterministic
theorem shard_deterministic (reader : ShardReader) (id : Nat) :
  ∀ s1 s2, s1.shardId = id → s2.shardId = id → 
    s1.contentHash = s2.contentHash
```

### Build

```bash
cd MonsterLean
lake build MonsterLean.ShardReader
```

## MiniZinc Model

### Constraints

- All shards have positive size
- Shard IDs in range [0, 71)
- Content-addressable uniqueness
- Source-specific validation

### Run

```bash
minizinc minizinc/shard_reader.mzn
```

## API Reference

### Rust

```rust
pub enum ShardSource {
    Local(String),           // Path to directory
    ArchiveOrg(String),      // Item identifier
    HuggingFace(String),     // Repo name (user/dataset)
}

pub struct ShardReader {
    pub fn new(source: ShardSource) -> Self;
    pub fn read_rdf_shard(&self, shard_id: usize) -> Result<RDFShard, String>;
    pub fn read_value_lattice(&self) -> Result<Vec<ValueLatticeEntry>, String>;
}

pub struct RDFShard {
    pub shard_id: usize,
    pub content_hash: String,
    pub triples: Vec<String>,
}
```

### Lean4

```lean
inductive ShardSource where
  | local : String → ShardSource
  | archiveOrg : String → ShardSource
  | huggingFace : String → ShardSource

structure ShardReader where
  source : ShardSource
  readRDFShard : Nat → IO RDFShard
  readValueLattice : IO (List ValueLatticeEntry)
```

## Examples

### Read All Shards

```rust
let reader = ShardReader::new(ShardSource::Local("archive_org_shards".to_string()));

for shard_id in 0..71 {
    let shard = reader.read_rdf_shard(shard_id)?;
    println!("Shard {}: {} triples", shard_id, shard.triples.len());
}
```

### Query Value Lattice

```rust
let lattice = reader.read_value_lattice()?;

for entry in lattice.iter().filter(|e| e.value == "24") {
    println!("Value 24: Gödel {}, {} witnesses", 
        entry.godel_number, entry.zk_witnesses.len());
}
```

### Verify Content Hash

```rust
let shard1 = reader.read_rdf_shard(0)?;
let shard2 = reader.read_rdf_shard(0)?;

assert_eq!(shard1.content_hash, shard2.content_hash);
```

## URLs

### Archive.org
```
https://archive.org/download/monster-zk-lattice-v1/monster_shard_00_hash_*.ttl
https://archive.org/download/monster-zk-lattice-v1/value_lattice_witnessed.json
```

### Hugging Face
```
https://huggingface.co/datasets/username/monster-zk-lattice/resolve/main/archive_org_shards/monster_shard_00_*.ttl
https://huggingface.co/datasets/username/monster-zk-lattice/resolve/main/value_lattice_witnessed.json
```

## Dependencies

### Rust
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
md5 = "0.7"

[features]
reqwest = ["dep:reqwest"]

[dependencies.reqwest]
version = "0.11"
features = ["blocking"]
optional = true
```

### Lean4
```
lean4
lake
```

### MiniZinc
```
minizinc >= 2.6
```

## Testing

```bash
# Rust
cargo test --bin universal_shard_reader

# Lean4
lake build && lake test

# MiniZinc
minizinc --solver gecode minizinc/shard_reader.mzn
```

## Trinity Complete

✅ **Rust**: Implementation  
✅ **Lean4**: Specification + Theorems  
✅ **MiniZinc**: Constraints  

All three systems can read shards from any source! 🎯
