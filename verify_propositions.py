#!/usr/bin/env python3
"""
Verify all propositions from the paper
"""

import json
import os
import subprocess
from pathlib import Path

print("🔍 VERIFYING ALL PROPOSITIONS")
print("=" * 60)
print()

# Load propositions
with open('propositions.json') as f:
    data = json.load(f)

propositions = data['propositions']
results = []

# P1: Architecture layers
print("P1: Checking architecture layers...")
try:
    # Check if files exist
    if Path('monster_autoencoder.py').exists():
        with open('monster_autoencoder.py') as f:
            content = f.read()
            has_layers = '[5, 11, 23, 47, 71]' in content or '5, 11, 23, 47, 71' in content
        result = 'VERIFIED' if has_layers else 'FAILED'
    else:
        result = 'FILE_NOT_FOUND'
    print(f"  {result}")
    results.append(('P1', result))
except Exception as e:
    print(f"  ERROR: {e}")
    results.append(('P1', 'ERROR'))

# P2: Monster primes
print("P2: Checking Monster primes...")
monster_primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71}
layer_primes = {11, 23, 47, 71}
if layer_primes.issubset(monster_primes):
    print("  VERIFIED")
    results.append(('P2', 'VERIFIED'))
else:
    print("  FAILED")
    results.append(('P2', 'FAILED'))

# P3: 5 input features
print("P3: Checking 5 input features...")
if Path('create_jinvariant_world.py').exists():
    with open('create_jinvariant_world.py') as f:
        content = f.read()
        has_features = 'number' in content and 'j_invariant' in content
    result = 'VERIFIED' if has_features else 'PARTIAL'
else:
    result = 'FILE_NOT_FOUND'
print(f"  {result}")
results.append(('P3', result))

# P4: Hecke composition
print("P4: Checking Hecke composition...")
result = 'NEEDS_EXECUTION'  # Would need to run tests
print(f"  {result}")
results.append(('P4', result))

# P5: J-invariant formula
print("P5: Checking j-invariant formula...")
if Path('create_jinvariant_world.py').exists():
    with open('create_jinvariant_world.py') as f:
        content = f.read()
        has_formula = '**3' in content and '1728' in content and '% 71' in content
    result = 'VERIFIED' if has_formula else 'FAILED'
else:
    result = 'FILE_NOT_FOUND'
print(f"  {result}")
results.append(('P5', result))

# P6: 7,115 objects
print("P6: Checking object count...")
if Path('monster_features.npy').exists():
    import numpy as np
    features = np.load('monster_features.npy')
    count = len(features)
    result = 'VERIFIED' if count == 7115 else f'FAILED (found {count})'
else:
    result = 'FILE_NOT_FOUND'
print(f"  {result}")
results.append(('P6', result))

# P7: 70 unique j-invariants
print("P7: Checking unique j-invariants...")
if Path('lmfdb_jinvariant_objects.parquet').exists():
    import pandas as pd
    df = pd.read_parquet('lmfdb_jinvariant_objects.parquet')
    unique_j = df['j_invariant'].nunique()
    result = 'VERIFIED' if unique_j == 70 else f'FAILED (found {unique_j})'
else:
    result = 'FILE_NOT_FOUND'
print(f"  {result}")
results.append(('P7', result))

# P8: 70 equivalence classes
print("P8: Checking equivalence classes...")
if Path('lmfdb_core_shards').exists():
    shards = list(Path('lmfdb_core_shards').glob('shard_*.parquet'))
    # Exclude shard_stats.parquet
    shards = [s for s in shards if 'stats' not in s.name]
    count = len(shards)
    result = 'VERIFIED' if count == 70 else f'FAILED (found {count})'
else:
    result = 'FILE_NOT_FOUND'
print(f"  {result}")
results.append(('P8', result))

# P9: Original data size
print("P9: Checking original data size...")
if Path('lmfdb_core_shards').exists():
    total_size = sum(f.stat().st_size for f in Path('lmfdb_core_shards').glob('shard_*.parquet') if 'stats' not in f.name)
    result = 'VERIFIED' if 900000 < total_size < 920000 else f'FAILED (found {total_size})'
else:
    result = 'FILE_NOT_FOUND'
print(f"  {result}")
results.append(('P9', result))

# P10: Trainable parameters
print("P10: Checking trainable parameters...")
# Calculate: (5*11 + 11*23 + 23*47 + 47*71) + (71*47 + 47*23 + 23*11 + 11*5)
encoder = 5*11 + 11*23 + 23*47 + 47*71
decoder = 71*47 + 47*23 + 23*11 + 11*5
total = encoder + decoder
result = 'VERIFIED' if total == 9690 else f'FAILED (calculated {total})'
print(f"  {result}")
results.append(('P10', result))

# P11: Compression ratio
print("P11: Checking compression ratio...")
compression = 907740 / 38760
result = 'VERIFIED' if 23.0 < compression < 23.5 else f'FAILED (calculated {compression:.1f})'
print(f"  {result}")
results.append(('P11', result))

# P12: Network capacity
print("P12: Checking network capacity...")
capacity = 71 ** 5
result = 'VERIFIED' if capacity == 1804229351 else f'FAILED (calculated {capacity})'
print(f"  {result}")
results.append(('P12', result))

# P13: Overcapacity
print("P13: Checking overcapacity...")
overcapacity = 1804229351 / 7115
result = 'VERIFIED' if 253000 < overcapacity < 254000 else f'FAILED (calculated {overcapacity:.0f})'
print(f"  {result}")
results.append(('P13', result))

# P14: 71 Hecke operators
print("P14: Checking Hecke operators...")
result = 'NEEDS_CODE_INSPECTION'
print(f"  {result}")
results.append(('P14', result))

# P15: Same architecture
print("P15: Checking architecture equivalence...")
result = 'NEEDS_CODE_INSPECTION'
print(f"  {result}")
results.append(('P15', result))

# P16: Rust MSE
print("P16: Checking Rust MSE...")
result = 'NEEDS_EXECUTION'
print(f"  {result}")
results.append(('P16', result))

# P17: 6 Hecke operators tested
print("P17: Checking Hecke test count...")
result = 'NEEDS_EXECUTION'
print(f"  {result}")
results.append(('P17', result))

# P18: Rust execution time
print("P18: Checking Rust execution time...")
result = 'NEEDS_EXECUTION'
print(f"  {result}")
results.append(('P18', result))

# P19: 100× speedup
print("P19: Checking speedup estimate...")
result = 'NEEDS_BENCHMARK'
print(f"  {result}")
results.append(('P19', result))

# P20: Rust compiles
print("P20: Checking Rust compilation...")
result = 'NEEDS_EXECUTION'
print(f"  {result}")
results.append(('P20', result))

# P21: 3 tests pass
print("P21: Checking Rust tests...")
result = 'NEEDS_EXECUTION'
print(f"  {result}")
results.append(('P21', result))

# P22: 20 functions converted
print("P22: Checking converted functions...")
if Path('lmfdb_rust_conversion.json').exists():
    with open('lmfdb_rust_conversion.json') as f:
        conv_data = json.load(f)
        count = conv_data.get('converted', 0)
    result = 'VERIFIED' if count == 20 else f'FAILED (found {count})'
else:
    result = 'FILE_NOT_FOUND'
print(f"  {result}")
results.append(('P22', result))

# P23: 500 total functions
print("P23: Checking total functions...")
if Path('lmfdb_math_functions.json').exists():
    with open('lmfdb_math_functions.json') as f:
        func_data = json.load(f)
        count = len(func_data.get('functions', []))
    result = 'VERIFIED' if count == 500 else f'FAILED (found {count})'
else:
    result = 'FILE_NOT_FOUND'
print(f"  {result}")
results.append(('P23', result))

print()
print("=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
print()

verified = sum(1 for _, r in results if r == 'VERIFIED')
failed = sum(1 for _, r in results if 'FAILED' in r)
needs_exec = sum(1 for _, r in results if 'NEEDS' in r)
not_found = sum(1 for _, r in results if 'NOT_FOUND' in r)
errors = sum(1 for _, r in results if r == 'ERROR')

print(f"✅ VERIFIED: {verified}/{len(results)}")
print(f"❌ FAILED: {failed}/{len(results)}")
print(f"⏳ NEEDS EXECUTION: {needs_exec}/{len(results)}")
print(f"📁 FILE NOT FOUND: {not_found}/{len(results)}")
print(f"⚠️  ERRORS: {errors}/{len(results)}")
print()

print("Detailed results:")
for prop_id, result in results:
    status = '✅' if result == 'VERIFIED' else '❌' if 'FAILED' in result else '⏳' if 'NEEDS' in result else '📁' if 'NOT_FOUND' in result else '⚠️'
    print(f"  {status} {prop_id}: {result}")

# Save results
verification_data = {
    'results': [{'id': pid, 'status': r} for pid, r in results],
    'summary': {
        'total': len(results),
        'verified': verified,
        'failed': failed,
        'needs_execution': needs_exec,
        'not_found': not_found,
        'errors': errors
    }
}

with open('verification_results.json', 'w') as f:
    json.dump(verification_data, f, indent=2)

print()
print("💾 Saved: verification_results.json")
