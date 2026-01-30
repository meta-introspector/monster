# Search Capabilities Across All Monster Systems

## 🔍 Primary Search Tools (Rust)

### 1. **expert_system.rs** ⭐⭐⭐
**Main search engine** - Searches both codebases with 710 keywords
```rust
fn search_keyword(&self, keyword: &str) -> Vec<FileMatch>
```
- Searches Monster codebase: `/home/mdupont/experiments/monster/`
- Searches zkPrologML: `/mnt/data1/nix/vendor/rust/github/`
- 710 keywords (71 per area × 10 areas)
- Multi-language: .nix, .rs, .lean, .pl, .mzn, .v

### 2. **generate_keywords.rs**
Generates search predicates for Prolog
```rust
search_group_1(Keyword) :- ...
```
- Creates searchable keyword database
- Prolog search predicates

### 3. **multi_lang_prover.rs**
Prolog search integration
```rust
fn prolog_search(&self, term: &str) -> Option<Vec<String>>
```
- Searches Prolog knowledge base
- Returns matching terms

### 4. **precedence_survey.rs**
Searches for .lean and .hlean files
```rust
.filter(|e| e.path().extension().map_or(false, |ext| ext == "hlean"))
```
- Surveys Lean4 files
- Finds precedence patterns

### 5. **universal_shard_reader.rs**
Searches across all 71 shards
```rust
.filter(|e| { ... })
```
- Reads from all shards
- Filters RDF data

### 6. **extract_constants.rs**
Searches for constants by pattern
```rust
.filter(|c| c.name.contains(pattern) || c.value.contains(pattern))
```
- Searches constant names
- Searches constant values

### 7. **term_frequency.rs**
Searches for term frequencies
```rust
.filter(|e| { ... })
```
- Counts term occurrences
- Frequency analysis

### 8. **virtual_knuth.rs**
Pattern search in text
```rust
text.find(pattern).and_then(|pos| { ... })
```
- Finds patterns in Lean code
- Searches for definitions

### 9. **llm_strip_miner.rs**
Searches LLM outputs
```rust
.find(|m| m.shard == query.shard && m.layer == query.layer)
```
- Searches by shard and layer
- Finds specific outputs

### 10. **mathematical_object_lattice.rs**
Searches mathematical objects
```rust
.filter(|o| o.object_type == obj_type)
.filter(|o| o.name.starts_with("Umbral"))
```
- Searches by object type
- Searches by name prefix

## 🔍 Data Search Tools

### 11. **qwen_strip_miner.rs**
```rust
.filter(|s| !s.is_empty() && s.ends_with(".parquet"))
```
- Searches Qwen parquet files

### 12. **vectorize_all_parquets.rs**
```rust
.filter(|s| !s.is_empty() && s.ends_with(".parquet"))
```
- Searches for parquet files

### 13. **parquet_gpu_pipeline.rs**
```rust
.filter(|e| e.path().extension().map_or(false, |ext| ext == "parquet"))
```
- Searches parquet files for GPU

### 14. **extract_71_objects.rs**
```rust
.filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
```
- Searches for Rust files

## 🔍 Review & Analysis Search

### 15. **review_paper.rs**
```rust
.filter(|e| e.as_ref().unwrap().path().extension().unwrap_or_default() == "png")
.filter(|r| r.role == "scholar")
.filter(|r| r.role == "muse")
```
- Searches for images
- Filters reviews by role

### 16. **pre_commit_review.rs**
```rust
response.find("SCORE:")
response.find("COMMENT:")
```
- Searches for score in review
- Searches for comments

### 17. **self_referential_conformal.rs**
```rust
.filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("c") || ...)
```
- Searches C files

## 🔍 GPU Search Tools

### 18. **cuda_unified_pipeline.rs**
```rust
.filter(|&&c| c > 0)
.filter(|m| m.shard == shard_id)
```
- Searches CUDA results
- Filters by shard

### 19. **cuda_monster_pipeline.rs**
```rust
.filter(|&&c| c > 0)
.filter(|m| m.shard == shard_id)
```
- Searches Monster CUDA results

### 20. **cuda_markov_bitwise.rs**
```rust
.filter(|m| m.shard == shard_id)
.filter(|&&c| c > 0)
```
- Searches Markov CUDA results

## 🔍 Shard Search Tools

### 21. **quantum_71_shards.rs**
```rust
format!("search_shard_{}", i)
```
- Creates search capability per shard
- 71 search endpoints

## 📊 Search Capabilities Summary

| Tool | Search Type | Target | Keywords |
|------|-------------|--------|----------|
| expert_system.rs | Keyword | Both codebases | 710 |
| multi_lang_prover.rs | Prolog | Knowledge base | Dynamic |
| precedence_survey.rs | File extension | .lean files | N/A |
| universal_shard_reader.rs | Shard | All 71 shards | N/A |
| extract_constants.rs | Pattern | Constants | User-defined |
| term_frequency.rs | Term | Text corpus | N/A |
| virtual_knuth.rs | Pattern | Lean code | User-defined |
| llm_strip_miner.rs | Shard+Layer | LLM outputs | N/A |
| mathematical_object_lattice.rs | Type+Name | Math objects | N/A |
| qwen_strip_miner.rs | File | Parquet files | N/A |

## 🎯 Best Search Tool by Use Case

**General code search:** `expert_system.rs` (710 keywords, both codebases)
**Prolog search:** `multi_lang_prover.rs`
**File search:** `precedence_survey.rs`, `qwen_strip_miner.rs`
**Shard search:** `universal_shard_reader.rs`, `quantum_71_shards.rs`
**Pattern search:** `extract_constants.rs`, `virtual_knuth.rs`
**Data search:** `llm_strip_miner.rs`, `mathematical_object_lattice.rs`

## 🔧 Add to Kiro Manifest

```json
{
  "name": "search_all",
  "description": "Search across all Monster systems: codebases (710 keywords), shards (71), files, patterns, data",
  "command": "cargo run --release --bin expert_system",
  "category": "search",
  "inputs": ["query", "scope"],
  "outputs": ["search_results"],
  "capabilities": [
    "keyword_search_710",
    "codebase_search_2",
    "shard_search_71",
    "file_search",
    "pattern_search",
    "prolog_search",
    "data_search"
  ]
}
```

## ∞ Total Search Capabilities

- **21 search tools** in Rust
- **710 keywords** (expert_system)
- **71 shards** (quantum_71_shards, universal_shard_reader)
- **2 codebases** (Monster + zkPrologML)
- **Multiple languages** (.rs, .lean, .pl, .nix, .v, .mzn, .parquet)

**∞ QED ∞**
