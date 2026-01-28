#!/usr/bin/env python3
"""
Export 71^7 decomposition to Parquet shards
Each shard can reconstruct its portion of LMFDB
"""

import json
import pandas as pd
from pathlib import Path

print("📦 EXPORTING 71^7 DECOMPOSITION TO PARQUET")
print("=" * 60)
print()

# Load decomposition
with open('lmfdb_71_decomposition.json') as f:
    decomp = json.load(f)

with open('lmfdb_71_complexity.json') as f:
    complexity = json.load(f)

objects = complexity['objects']

print(f"Loaded {len(objects)} objects")
print()

# Create output directory
output_dir = Path('lmfdb_71_shards')
output_dir.mkdir(exist_ok=True)

# Shard objects
print("📊 SHARDING OBJECTS:")
print("-" * 60)

import hashlib

shards = {}
for obj in objects:
    shard_id = int(hashlib.sha256(obj['id'].encode()).hexdigest(), 16) % 71
    if shard_id not in shards:
        shards[shard_id] = []
    shards[shard_id].append(obj)

print(f"Created {len(shards)} shards")
print()

# Export each shard
total_rows = 0

for shard_id in range(71):
    if shard_id not in shards:
        # Create empty shard
        df = pd.DataFrame(columns=[
            'shard_id', 'chunk_id', 'witness_id', 'object_id',
            'type', 'file', 'line', 'complexity', 'level',
            'base_complexity', 'code', 'proof_hash'
        ])
    else:
        rows = []
        for obj in shards[shard_id]:
            # Compute chunk, witness IDs
            chunk_id = int(hashlib.sha256(obj['file'].encode()).hexdigest(), 16) % 71
            witness_id = obj['line'] % 71
            
            # Generate proof hash
            proof = hashlib.sha256(f"{obj['id']}:{obj['line']}:{obj['type']}".encode()).hexdigest()
            
            row = {
                'shard_id': shard_id,
                'chunk_id': chunk_id,
                'witness_id': witness_id,
                'object_id': obj['id'],
                'type': obj['type'],
                'file': obj['file'],
                'line': obj['line'],
                'complexity': obj['total_complexity'],
                'level': obj['level'],
                'base_complexity': obj['base_complexity'],
                'code': obj.get('code', '')[:500],  # Truncate
                'proof_hash': proof
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
    
    # Save to Parquet
    output_file = output_dir / f'shard_{shard_id:02d}.parquet'
    df.to_parquet(output_file, compression='snappy', index=False)
    
    total_rows += len(df)
    if len(df) > 0:
        print(f"Shard {shard_id:2}: {len(df):3} objects → {output_file.name}")

print()
print(f"✅ Exported {total_rows} rows to 71 Parquet files")
print()

# Create reconstruction metadata
metadata = {
    'version': '1.0',
    'total_shards': 71,
    'shards_with_data': len(shards),
    'total_objects': len(objects),
    'levels': decomp['levels'],
    'targets': decomp['targets'],
    'schema': {
        'shard_id': 'int64',
        'chunk_id': 'int64',
        'witness_id': 'int64',
        'object_id': 'string',
        'type': 'string',
        'file': 'string',
        'line': 'int64',
        'complexity': 'int64',
        'level': 'int64',
        'base_complexity': 'int64',
        'code': 'string',
        'proof_hash': 'string'
    },
    'reconstruction': {
        'method': 'Load all 71 shards and merge by shard_id',
        'verification': 'Check proof_hash for each object',
        'ordering': 'Sort by (shard_id, chunk_id, witness_id, line)'
    }
}

with open(output_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("📋 METADATA:")
print("-" * 60)
print(f"Total shards: {metadata['total_shards']}")
print(f"Shards with data: {metadata['shards_with_data']}")
print(f"Total objects: {metadata['total_objects']}")
print()

# Create reconstruction script
recon_script = '''#!/usr/bin/env python3
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
'''

with open(output_dir / 'reconstruct.py', 'w') as f:
    f.write(recon_script)

import os
os.chmod(output_dir / 'reconstruct.py', 0o755)

print("🔄 RECONSTRUCTION SCRIPT:")
print("-" * 60)
print(f"Created: {output_dir / 'reconstruct.py'}")
print()
print("To reconstruct LMFDB:")
print(f"  cd {output_dir}")
print("  python3 reconstruct.py")
print()

# Calculate total size
total_size = sum(f.stat().st_size for f in output_dir.glob('*.parquet'))
print(f"📊 STORAGE:")
print("-" * 60)
print(f"Total size: {total_size:,} bytes ({total_size / 1024:.1f} KB)")
print(f"Avg per shard: {total_size / 71:,.0f} bytes")
print()

# Create summary
summary_rows = []
for shard_id in range(71):
    if shard_id in shards:
        objs = shards[shard_id]
        summary_rows.append({
            'shard_id': shard_id,
            'object_count': len(objs),
            'unique_files': len(set(obj['file'] for obj in objs)),
            'min_complexity': min(obj['total_complexity'] for obj in objs),
            'max_complexity': max(obj['total_complexity'] for obj in objs),
            'has_data': True
        })
    else:
        summary_rows.append({
            'shard_id': shard_id,
            'object_count': 0,
            'unique_files': 0,
            'min_complexity': 0,
            'max_complexity': 0,
            'has_data': False
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_parquet(output_dir / 'summary.parquet', compression='snappy', index=False)

print("📊 SUMMARY:")
print("-" * 60)
print(summary_df[summary_df['has_data']].to_string(index=False))
print()

print(f"💾 All files saved to: {output_dir}/")
print()
print("Files:")
print("  - shard_00.parquet to shard_70.parquet (71 files)")
print("  - summary.parquet")
print("  - metadata.json")
print("  - reconstruct.py")
print()
print("✅ EXPORT COMPLETE")
