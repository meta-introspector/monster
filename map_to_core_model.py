#!/usr/bin/env python3
"""
Map all items to core Monster model and shard by complexity
- Constants → Level → Shard
- Functions → Level → Shard
- Math functions → Level → Shard
All sharded by complexity mod 71
"""

import json
import hashlib
from collections import defaultdict
import pandas as pd
from pathlib import Path

print("🔮 MAPPING TO CORE MODEL & SHARDING BY COMPLEXITY")
print("=" * 60)
print()

# Load all data
print("Loading data...")
with open('lmfdb_ast_analysis.json') as f:
    ast_data = json.load(f)

with open('lmfdb_math_functions.json') as f:
    math_data = json.load(f)

with open('lmfdb_71_complexity.json') as f:
    complexity_data = json.load(f)

print("✓ Loaded AST analysis")
print("✓ Loaded math functions")
print("✓ Loaded complexity data")
print()

# Core model structure
core_model = {
    'constants': [],
    'functions': [],
    'math_functions': [],
    'objects': []
}

# Map constants
print("Mapping constants to core model...")
for const in ast_data['constants'][:1000]:  # Limited sample
    item = {
        'id': hashlib.sha256(f"{const['file']}:{const['line']}".encode()).hexdigest()[:8],
        'type': 'constant',
        'subtype': const['value_type'],
        'value': const['value'][:50],
        'file': const['file'],
        'line': const['line'],
        'complexity': const['complexity'],
        'level': const['level'],
        'shard': const['level'] % 71,  # Shard by level
        'has_71': const['has_71']
    }
    core_model['constants'].append(item)

print(f"✓ Mapped {len(core_model['constants'])} constants")

# Map functions
print("Mapping functions to core model...")
for func in ast_data['functions']:
    item = {
        'id': hashlib.sha256(f"{func['file']}:{func['line']}:{func['name']}".encode()).hexdigest()[:8],
        'type': 'function',
        'name': func['name'],
        'file': func['file'],
        'line': func['line'],
        'num_statements': func['num_statements'],
        'num_args': func['num_args'],
        'complexity': func['complexity'],
        'level': func['level'],
        'shard': func['level'] % 71,
        'has_71': func['has_71']
    }
    core_model['functions'].append(item)

print(f"✓ Mapped {len(core_model['functions'])} functions")

# Map math functions
print("Mapping math functions to core model...")
for func in math_data['functions']:
    item = {
        'id': hashlib.sha256(f"{func['file']}:{func['line']}:{func['name']}".encode()).hexdigest()[:8],
        'type': 'math_function',
        'name': func['name'],
        'file': func['file'],
        'line': func['line'],
        'num_args': func['num_args'],
        'num_binops': func['num_binops'],
        'num_calls': func['num_calls'],
        'cyclomatic_complexity': func['cyclomatic_complexity'],
        'math_complexity': func['math_complexity'],
        'level': func['level'],
        'shard': func['level'] % 71,
        'has_71': func['has_71']
    }
    core_model['math_functions'].append(item)

print(f"✓ Mapped {len(core_model['math_functions'])} math functions")

# Map original objects
print("Mapping original objects to core model...")
for obj in complexity_data['objects']:
    item = {
        'id': obj['id'],
        'type': 'object',
        'subtype': obj['type'],
        'file': obj['file'],
        'line': obj['line'],
        'complexity': obj['total_complexity'],
        'level': obj['level'],
        'shard': obj['level'] % 71,
        'base_complexity': obj['base_complexity']
    }
    core_model['objects'].append(item)

print(f"✓ Mapped {len(core_model['objects'])} objects")
print()

# Combine all items
all_items = (
    core_model['constants'] +
    core_model['functions'] +
    core_model['math_functions'] +
    core_model['objects']
)

print(f"Total items in core model: {len(all_items)}")
print()

# Shard by complexity
print("📦 SHARDING BY COMPLEXITY (MOD 71):")
print("-" * 60)

shards = defaultdict(list)
for item in all_items:
    shard_id = item['shard']
    shards[shard_id].append(item)

print(f"Created {len(shards)} shards")
print()

# Show shard distribution
print("Shard distribution:")
for shard_id in sorted(shards.keys())[:20]:
    items = shards[shard_id]
    types = defaultdict(int)
    for item in items:
        types[item['type']] += 1
    
    print(f"Shard {shard_id:2}: {len(items):5} items "
          f"(C:{types['constant']:4}, F:{types['function']:4}, "
          f"M:{types['math_function']:3}, O:{types['object']:2})")

print()

# Statistics by shard
print("📊 SHARD STATISTICS:")
print("-" * 60)

shard_stats = []
for shard_id in range(71):
    if shard_id in shards:
        items = shards[shard_id]
        complexities = [i.get('complexity', i.get('math_complexity', 0)) for i in items]
        stats = {
            'shard_id': shard_id,
            'total_items': len(items),
            'constants': len([i for i in items if i['type'] == 'constant']),
            'functions': len([i for i in items if i['type'] == 'function']),
            'math_functions': len([i for i in items if i['type'] == 'math_function']),
            'objects': len([i for i in items if i['type'] == 'object']),
            'with_71': len([i for i in items if i.get('has_71', False)]),
            'avg_complexity': sum(complexities) / len(complexities) if complexities else 0,
            'max_complexity': max(complexities) if complexities else 0,
            'min_complexity': min(complexities) if complexities else 0
        }
    else:
        stats = {
            'shard_id': shard_id,
            'total_items': 0,
            'constants': 0,
            'functions': 0,
            'math_functions': 0,
            'objects': 0,
            'with_71': 0,
            'avg_complexity': 0,
            'max_complexity': 0,
            'min_complexity': 0
        }
    shard_stats.append(stats)

# Top 10 shards by item count
top_shards = sorted(shard_stats, key=lambda x: -x['total_items'])[:10]
print("Top 10 shards by item count:")
for stats in top_shards:
    print(f"Shard {stats['shard_id']:2}: {stats['total_items']:5} items, "
          f"avg complexity {stats['avg_complexity']:.1f}")

print()

# Export to Parquet
print("📦 EXPORTING TO PARQUET:")
print("-" * 60)

output_dir = Path('lmfdb_core_shards')
output_dir.mkdir(exist_ok=True)

for shard_id in range(71):
    if shard_id in shards:
        df = pd.DataFrame(shards[shard_id])
    else:
        # Empty shard
        df = pd.DataFrame(columns=['id', 'type', 'file', 'line', 'complexity', 'level', 'shard'])
    
    output_file = output_dir / f'shard_{shard_id:02d}.parquet'
    df.to_parquet(output_file, compression='snappy', index=False)
    
    if len(df) > 0:
        print(f"Shard {shard_id:2}: {len(df):5} items → {output_file.name}")

print()
print(f"✅ Exported {len(shards)} shards to Parquet")
print()

# Save shard statistics
stats_df = pd.DataFrame(shard_stats)
stats_df.to_parquet(output_dir / 'shard_stats.parquet', compression='snappy', index=False)

print("📊 Shard statistics:")
print(stats_df.describe())
print()

# Save core model
with open('lmfdb_core_model.json', 'w') as f:
    json.dump({
        'total_items': len(all_items),
        'by_type': {
            'constants': len(core_model['constants']),
            'functions': len(core_model['functions']),
            'math_functions': len(core_model['math_functions']),
            'objects': len(core_model['objects'])
        },
        'shards': len(shards),
        'items_per_shard': {str(k): len(v) for k, v in shards.items()},
        'shard_stats': shard_stats
    }, f, indent=2)

print(f"💾 Saved core model: lmfdb_core_model.json")
print()

# Create reconstruction script
recon_script = '''#!/usr/bin/env python3
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
    print(f"\\nReconstructed {len(lmfdb)} items")
    print(f"Types: {lmfdb['type'].value_counts().to_dict()}")
    lmfdb.to_parquet('lmfdb_core_reconstructed.parquet', compression='snappy', index=False)
    print("✅ Saved to: lmfdb_core_reconstructed.parquet")
'''

with open(output_dir / 'reconstruct.py', 'w') as f:
    f.write(recon_script)

import os
os.chmod(output_dir / 'reconstruct.py', 0o755)

print(f"🔄 Created reconstruction script: {output_dir / 'reconstruct.py'}")
print()

# Summary
print("=" * 60)
print("CORE MODEL SUMMARY")
print("=" * 60)
print()
print(f"Total items: {len(all_items):,}")
print(f"  Constants: {len(core_model['constants']):,}")
print(f"  Functions: {len(core_model['functions']):,}")
print(f"  Math functions: {len(core_model['math_functions']):,}")
print(f"  Objects: {len(core_model['objects']):,}")
print()
print(f"Shards: {len(shards)}/71")
print(f"Items per shard (avg): {len(all_items) / len(shards):.1f}")
print(f"Items per shard (max): {max(len(v) for v in shards.values())}")
print(f"Items per shard (min): {min(len(v) for v in shards.values())}")
print()
print("✅ CORE MODEL MAPPING COMPLETE")
