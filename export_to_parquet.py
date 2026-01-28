#!/usr/bin/env python3
"""
Export LMFDB Hecke shards to Parquet format
Each shard becomes a separate Parquet file
"""

import json
import pandas as pd
from pathlib import Path

print("📦 EXPORTING LMFDB HECKE SHARDS TO PARQUET")
print("=" * 60)
print()

# Load the 71 shards
with open('lmfdb_71_shards.json') as f:
    data = json.load(f)

shards = data['shards']
stats = data['stats']

print(f"Loaded {stats['total_chunks']} chunks in {len(shards)} shards")
print()

# Create output directory
output_dir = Path('lmfdb_parquet_shards')
output_dir.mkdir(exist_ok=True)

print("📊 EXPORTING SHARDS:")
print("-" * 60)

total_rows = 0

for shard_id, chunks in shards.items():
    # Convert to DataFrame
    rows = []
    for chunk in chunks:
        row = {
            'shard_id': int(shard_id),
            'chunk_name': chunk['name'],
            'chunk_type': chunk['type'],
            'file': chunk['file'],
            'line_start': chunk['line_start'],
            'line_end': chunk['line_end'],
            'lines': chunk['lines'],
            'bytes': chunk['bytes'],
            'has_71': chunk['has_71'],
            'hash': chunk.get('hash', ''),
            'code': chunk.get('code', '')[:1000] if chunk.get('code') else ''  # Truncate long code
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to Parquet
    shard_num = int(shard_id)
    output_file = output_dir / f'shard_{shard_num:02d}.parquet'
    df.to_parquet(output_file, compression='snappy', index=False)
    
    total_rows += len(rows)
    print(f"Shard {shard_num:2}: {len(rows):3} chunks → {output_file.name}")

print()
print(f"✅ Exported {total_rows} rows to {len(shards)} Parquet files")
print()

# Create metadata file
metadata = {
    'total_chunks': stats['total_chunks'],
    'total_shards': len(shards),
    'dominant_shard': stats['dominant_shard'],
    'dominant_count': stats['dominant_count'],
    'format': 'parquet',
    'compression': 'snappy',
    'schema': {
        'shard_id': 'int64',
        'chunk_name': 'string',
        'chunk_type': 'string',
        'file': 'string',
        'line_start': 'int64',
        'line_end': 'int64',
        'lines': 'int64',
        'bytes': 'int64',
        'has_71': 'bool',
        'hash': 'string',
        'code': 'string'
    }
}

with open(output_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("📋 METADATA:")
print("-" * 60)
print(f"Total chunks: {metadata['total_chunks']}")
print(f"Total shards: {metadata['total_shards']}")
print(f"Dominant shard: {metadata['dominant_shard']} ({metadata['dominant_count']} chunks)")
print(f"Format: Parquet (Snappy compression)")
print()

# Calculate total size
total_size = sum(f.stat().st_size for f in output_dir.glob('*.parquet'))
print(f"Total size: {total_size:,} bytes ({total_size / 1024:.1f} KB)")
print()

# Create summary DataFrame
summary_rows = []
for shard_id, chunks in shards.items():
    summary_rows.append({
        'shard_id': int(shard_id),
        'chunk_count': len(chunks),
        'total_bytes': sum(c['bytes'] for c in chunks),
        'has_monster_prime': any(c['has_71'] for c in chunks)
    })

summary_df = pd.DataFrame(summary_rows).sort_values('shard_id')
summary_df.to_parquet(output_dir / 'summary.parquet', compression='snappy', index=False)

print("📊 SUMMARY STATISTICS:")
print("-" * 60)
print(summary_df.to_string(index=False))
print()

print(f"💾 All files saved to: {output_dir}/")
print()
print("Files:")
print(f"  - shard_XX.parquet (71 files)")
print(f"  - summary.parquet")
print(f"  - metadata.json")
