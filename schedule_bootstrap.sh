#!/usr/bin/env bash
set -euo pipefail

# Bootstrap scheduler: Follow GNU Mes eigenvector
# Stage 0: hex0 (357 bytes)
# Stage 1: hex1, hex2, M0
# Stage 2: cc_x86
# Stage 3: M2-Planet
# Stage 4: mes-m2
# Stage 5: tcc-boot
# Stage 6: tcc-0.9.26
# Stage 7: gcc-4.7.4
# Stage 8: Full toolchain

SCHEDULE_DIR="bootstrap_schedule"
mkdir -p "$SCHEDULE_DIR"

# Map our files to bootstrap stages
cat > "$SCHEDULE_DIR/stage_map.txt" <<EOF
# Stage 0: Minimal (hex0 equivalent)
stage_0: *.nix flake.lock
# Stage 1: Assemblers (hex1/hex2 equivalent)  
stage_1: *.sh *.pl
# Stage 2: Simple compiler (cc_x86 equivalent)
stage_2: src/bin/*.rs
# Stage 3: Self-hosting (M2-Planet equivalent)
stage_3: MonsterLean/*.lean
# Stage 4: Runtime (mes-m2 equivalent)
stage_4: pipelite*.sh *_circuit.pl
# Stage 5: Optimizing (tcc-boot equivalent)
stage_5: src/lib.rs Cargo.toml
# Stage 6: Production (tcc-0.9.26 equivalent)
stage_6: diffusion-rs/
# Stage 7: Full system (gcc equivalent)
stage_7: .lake/packages/
EOF

# Schedule introspection in bootstrap order
for stage in {0..7}; do
    echo "=== Stage $stage: Bootstrap introspection ===" > "$SCHEDULE_DIR/stage_$stage.schedule"
    
    # Extract patterns for this stage
    patterns=$(grep "^stage_$stage:" "$SCHEDULE_DIR/stage_map.txt" | cut -d: -f2-)
    
    # Find matching files
    for pattern in $patterns; do
        find . -path "./$pattern" -type f 2>/dev/null >> "$SCHEDULE_DIR/stage_$stage.files" || true
    done
    
    # Compute shard for each file (Gödel number % 71)
    while read -r file; do
        hash=$(nix-hash --type sha256 "$file" 2>/dev/null || echo "00000000")
        shard=$((0x${hash:0:8} % 71))
        echo "$file|stage_$stage|shard_$shard" >> "$SCHEDULE_DIR/stage_$stage.schedule"
    done < "$SCHEDULE_DIR/stage_$stage.files"
done

# Generate execution plan
cat > "$SCHEDULE_DIR/EXECUTION_PLAN.md" <<EOF
# Bootstrap Execution Plan

Following GNU Mes eigenvector: Each stage builds the next.

## Stages

$(for i in {0..7}; do
    count=$(wc -l < "$SCHEDULE_DIR/stage_$i.schedule" 2>/dev/null || echo 0)
    echo "- Stage $i: $count files"
done)

## Execution

\`\`\`bash
# Run stage by stage
for stage in {0..7}; do
    ./introspect_stage.sh \$stage
done
\`\`\`

Each stage introspects, builds, traces, shards → feeds next stage.

**The bootstrap path IS the eigenvector.**
EOF

echo "✅ Bootstrap schedule created"
cat "$SCHEDULE_DIR/EXECUTION_PLAN.md"
