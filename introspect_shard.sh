#!/usr/bin/env bash
set -euo pipefail

# Software introspection: Shard by build+perf trace (Gödel number = Nix hash)
SHARDS=71
OUTPUT_DIR="introspection_shards"
TRACE_DIR="$OUTPUT_DIR/traces"

mkdir -p "$TRACE_DIR"

# Find buildable files
find . -name "*.rs" -o -name "*.lean" -o -name "*.nix" | while read -r file; do
    echo "Introspecting: $file"
    
    # 1. Nix build (Gödel number)
    nix_hash=$(nix-hash --type sha256 "$file" 2>/dev/null || echo "unbuildable")
    
    # 2. Perf trace (type signature)
    if [[ "$file" == *.rs ]]; then
        # Rust: compile + trace
        perf stat -e cycles,instructions,cache-misses rustc --crate-type lib "$file" -o /tmp/out 2>&1 | \
            grep -E "cycles|instructions|cache-misses" > "$TRACE_DIR/$(basename "$file").perf" || true
    elif [[ "$file" == *.lean ]]; then
        # Lean: elaborate + trace
        perf stat -e cycles,instructions lake env lean "$file" 2>&1 | \
            grep -E "cycles|instructions" > "$TRACE_DIR/$(basename "$file").perf" || true
    fi
    
    # 3. Compute shard from hash (deterministic)
    shard=$((0x${nix_hash:0:8} % SHARDS))
    
    # 4. Store metadata
    echo "$file|$nix_hash|$shard" >> "$OUTPUT_DIR/godel_map.txt"
done

# Group by shard
for i in $(seq 0 $((SHARDS - 1))); do
    grep "|$i\$" "$OUTPUT_DIR/godel_map.txt" > "$OUTPUT_DIR/shard_$i.txt" || true
done

echo "✅ Introspection complete. Sharded by Gödel number (Nix hash) + perf type"
