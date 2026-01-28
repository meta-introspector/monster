#!/usr/bin/env python3
"""
Reconstruct LMFDB from 71 Parquet shards
"""

import pandas as pd
from pathlib import Path

print("🔄 RECONSTRUCTING LMFDB FROM 71 SHARDS")
print("=" * 60)
print()

shard_dir = Path('lmfdb_71_shards')

# Load all shards
shards = []
for i in range(71):
    shard_file = shard_dir / f'shard_{i:02d}.parquet'
    if shard_file.exists():
        df = pd.read_parquet(shard_file)
        if len(df) > 0:
            shards.append(df)
            print(f"Loaded shard {i:2}: {len(df)} objects")

print()
print(f"Loaded {len(shards)} shards")
print()

# Merge all shards
if shards:
    lmfdb = pd.concat(shards, ignore_index=True)
    
    # Sort by hierarchy
    lmfdb = lmfdb.sort_values(['shard_id', 'chunk_id', 'witness_id', 'line'])
    
    print("📊 RECONSTRUCTED LMFDB:")
    print("-" * 60)
    print(f"Total objects: {len(lmfdb)}")
    print(f"Unique files: {lmfdb['file'].nunique()}")
    print(f"Unique types: {lmfdb['type'].nunique()}")
    print(f"Complexity range: {lmfdb['complexity'].min()}-{lmfdb['complexity'].max()}")
    print()
    
    # Verify proofs
    print("🔐 VERIFYING PROOFS:")
    print("-" * 60)
    
    import hashlib
    
    verified = 0
    for _, row in lmfdb.iterrows():
        expected = hashlib.sha256(
            f"{row['object_id']}:{row['line']}:{row['type']}".encode()
        ).hexdigest()
        if expected == row['proof_hash']:
            verified += 1
    
    print(f"Verified: {verified}/{len(lmfdb)} objects ({100*verified/len(lmfdb):.1f}%)")
    print()
    
    # Save reconstructed LMFDB
    lmfdb.to_parquet('lmfdb_reconstructed.parquet', compression='snappy', index=False)
    print("💾 Saved to: lmfdb_reconstructed.parquet")
    print()
    
    # Show sample
    print("📋 SAMPLE OBJECTS:")
    print("-" * 60)
    print(lmfdb[['shard_id', 'type', 'file', 'line', 'complexity']].head(10).to_string(index=False))
    print()
    
    print("✅ RECONSTRUCTION COMPLETE")
else:
    print("❌ No shards found")
