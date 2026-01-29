#!/usr/bin/env bash
set -euo pipefail

# Shard all files into 71 directories (largest Monster prime)
SHARDS=71
OUTPUT_DIR="shards"

mkdir -p "$OUTPUT_DIR"

# Initialize shard directories
for i in $(seq 0 $((SHARDS - 1))); do
    mkdir -p "$OUTPUT_DIR/shard_$(printf "%02d" $i)"
done

# Find all files (excluding .git, shards dir itself, and common build artifacts)
echo "Finding files..."
find . -type f \
    ! -path "./.git/*" \
    ! -path "./shards/*" \
    ! -path "./target/*" \
    ! -path "./.lake/*" \
    ! -path "./node_modules/*" \
    -print0 | while IFS= read -r -d '' file; do
    
    # Hash filename to determine shard (deterministic)
    hash=$(echo -n "$file" | md5sum | cut -d' ' -f1)
    # Convert hex to decimal and mod by 71
    shard=$((0x${hash:0:8} % SHARDS))
    
    # Create symlink in shard directory
    shard_dir="$OUTPUT_DIR/shard_$(printf "%02d" $shard)"
    # Remove leading ./
    clean_path="${file#./}"
    # Create parent dirs in shard
    mkdir -p "$shard_dir/$(dirname "$clean_path")"
    # Symlink to original
    ln -sf "$(realpath "$file")" "$shard_dir/$clean_path"
done

# Generate shard manifest
echo "Generating manifest..."
for i in $(seq 0 $((SHARDS - 1))); do
    shard_dir="$OUTPUT_DIR/shard_$(printf "%02d" $i)"
    count=$(find "$shard_dir" -type l | wc -l)
    echo "Shard $i: $count files" >> "$OUTPUT_DIR/MANIFEST.txt"
done

echo "✅ Sharded into 71 directories"
cat "$OUTPUT_DIR/MANIFEST.txt"
