#!/usr/bin/env python3
"""Reconstruct LMFDB from core model shards"""
import pandas as pd
from pathlib import Path

shard_dir = Path('lmfdb_core_shards')
shards = []

for i in range(71):
    shard_file = shard_dir / f'shard_{i:02d}.parquet'
    if shard_file.exists():
        df = pd.read_parquet(shard_file)
        if len(df) > 0:
            shards.append(df)
            print(f"Loaded shard {i:2}: {len(df)} items")

if shards:
    lmfdb = pd.concat(shards, ignore_index=True)
    print(f"\nReconstructed {len(lmfdb)} items")
    print(f"Types: {lmfdb['type'].value_counts().to_dict()}")
    lmfdb.to_parquet('lmfdb_core_reconstructed.parquet', compression='snappy', index=False)
    print("✅ Saved to: lmfdb_core_reconstructed.parquet")
