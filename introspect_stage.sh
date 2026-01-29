#!/usr/bin/env bash
set -euo pipefail

STAGE=$1
SCHEDULE_DIR="bootstrap_schedule"
TRACE_DIR="$SCHEDULE_DIR/traces/stage_$STAGE"
mkdir -p "$TRACE_DIR"

echo "🔍 Introspecting Stage $STAGE..."

# Read schedule
while IFS='|' read -r file stage shard; do
    echo "  Building: $file → $shard"
    
    # Nix build (Gödel number)
    nix_hash=$(nix-hash --type sha256 "$file" 2>/dev/null || echo "unbuildable")
    
    # Perf trace (type)
    perf stat -e cycles,instructions,cache-misses cat "$file" > /dev/null 2> "$TRACE_DIR/$(basename "$file").perf" || true
    
    # Record
    echo "$file|$nix_hash|$shard|stage_$STAGE" >> "$SCHEDULE_DIR/introspection_results.txt"
done < "$SCHEDULE_DIR/stage_$STAGE.schedule"

echo "✅ Stage $STAGE complete"
