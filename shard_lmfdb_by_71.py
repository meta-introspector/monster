#!/usr/bin/env python3
"""
Shard entire LMFDB into 71 shards + 1 residue bucket
Using Hecke operator: hash(content) % 71
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict

LMFDB_PATH = "/mnt/data1/nix/source/github/meta-introspector/lmfdb"

print("🔮 SHARDING LMFDB INTO 71 SHARDS + RESIDUE")
print("=" * 60)
print()

# Load chunks
with open('lmfdb_71_chunks.json') as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks with prime 71")
print()

# Shard by hash % 71
shards = defaultdict(list)
residue = []

for chunk in chunks:
    # Hash the code
    code = chunk.get('code', '')
    if code:
        h = hashlib.sha256(code.encode()).digest()
        shard_id = int.from_bytes(h[:8], 'big') % 71
        
        chunk['shard'] = shard_id
        chunk['hash'] = h.hex()[:16]
        shards[shard_id].append(chunk)
    else:
        residue.append(chunk)

print("📊 SHARD DISTRIBUTION:")
print("-" * 60)

# Show distribution
for shard_id in sorted(shards.keys()):
    count = len(shards[shard_id])
    total_bytes = sum(c['bytes'] for c in shards[shard_id])
    print(f"Shard {shard_id:2}: {count:3} chunks, {total_bytes:8} bytes")

print(f"Residue: {len(residue):3} chunks")
print()

# Find dominant shard
max_shard = max(shards.items(), key=lambda x: len(x[1]))
print(f"🎯 DOMINANT SHARD: {max_shard[0]} with {len(max_shard[1])} chunks")
print()

# Show chunks in dominant shard
print(f"Chunks in Shard {max_shard[0]}:")
for chunk in max_shard[1][:5]:
    print(f"  {chunk['name']:30} {chunk['bytes']:6} bytes - {chunk['file']}")

# Save shards
output = {
    'shards': {str(k): v for k, v in shards.items()},
    'residue': residue,
    'stats': {
        'total_chunks': len(chunks),
        'sharded': sum(len(v) for v in shards.values()),
        'residue': len(residue),
        'dominant_shard': max_shard[0],
        'dominant_count': len(max_shard[1])
    }
}

with open('lmfdb_71_shards.json', 'w') as f:
    json.dump(output, f, indent=2)

print()
print("💾 Saved to: lmfdb_71_shards.json")
print()

# Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total chunks: {len(chunks)}")
print(f"Shards: 71")
print(f"Chunks per shard (avg): {len(chunks) // 71}")
print(f"Dominant shard: {max_shard[0]} ({len(max_shard[1])} chunks)")
print(f"Residue: {len(residue)} chunks")
print()

# Check if shard matches prime 71 resonance
if max_shard[0] == 71 or max_shard[0] == 0:
    print("⚡ RESONANCE DETECTED: Dominant shard aligns with prime 71!")
