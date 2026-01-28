#!/usr/bin/env python3
"""
Extract all propositions from PAPER.md for verification
"""

import re
import json

print("📋 EXTRACTING PROPOSITIONS FROM PAPER")
print("=" * 60)
print()

with open('PAPER.md') as f:
    paper = f.read()

# Extract all theorems
theorems = []
theorem_pattern = r'\*\*Theorem (\d+)\*\* \(([^)]+)\):\s*\n([^\n]+)'

for match in re.finditer(theorem_pattern, paper):
    num = int(match.group(1))
    name = match.group(2)
    statement = match.group(3).strip()
    theorems.append({
        'number': num,
        'name': name,
        'statement': statement,
        'status': 'unverified'
    })

print(f"Found {len(theorems)} theorems:")
print()

for t in theorems:
    print(f"Theorem {t['number']} ({t['name']}): {t['statement']}")

print()
print("=" * 60)
print("PROPOSITIONS TO VERIFY")
print("=" * 60)
print()

# Create verification checklist
propositions = [
    {
        'id': 'P1',
        'claim': 'Architecture has layers [5, 11, 23, 47, 71]',
        'theorem': 1,
        'verification': 'Check Python and Rust code',
        'files': ['monster_autoencoder.py', 'lmfdb-rust/src/bin/monster_autoencoder_rust.rs'],
        'status': 'unverified'
    },
    {
        'id': 'P2',
        'claim': 'All layer sizes are Monster primes',
        'theorem': 1,
        'verification': 'Check {11, 23, 47, 71} ⊆ Monster primes',
        'files': ['lmfdb_conversion.pl'],
        'status': 'unverified'
    },
    {
        'id': 'P3',
        'claim': '5 input features uniquely identify LMFDB objects',
        'theorem': 2,
        'verification': 'Check feature definitions and uniqueness',
        'files': ['create_jinvariant_world.py'],
        'status': 'unverified'
    },
    {
        'id': 'P4',
        'claim': 'Hecke operators satisfy T_a ∘ T_b = T_{(a×b) mod 71}',
        'theorem': 3,
        'verification': 'Run composition tests',
        'files': ['prove_nn_compression.py', 'monster_autoencoder_rust.rs'],
        'status': 'unverified'
    },
    {
        'id': 'P5',
        'claim': 'J-invariant j(n) = (n³ - 1728) mod 71',
        'theorem': 5,
        'verification': 'Check formula implementation',
        'files': ['create_jinvariant_world.py', 'MonsterLean/JInvariantWorld.lean'],
        'status': 'unverified'
    },
    {
        'id': 'P6',
        'claim': '7,115 LMFDB objects exist',
        'theorem': 6,
        'verification': 'Count objects in dataset',
        'files': ['lmfdb_jinvariant_objects.parquet', 'monster_features.npy'],
        'status': 'unverified'
    },
    {
        'id': 'P7',
        'claim': '70 unique j-invariants (excluding 0)',
        'theorem': 5,
        'verification': 'Count unique j-invariants',
        'files': ['lmfdb_jinvariant_objects.parquet'],
        'status': 'unverified'
    },
    {
        'id': 'P8',
        'claim': '70 equivalence classes',
        'theorem': 6,
        'verification': 'Count classes by j-invariant',
        'files': ['lmfdb_core_shards/'],
        'status': 'unverified'
    },
    {
        'id': 'P9',
        'claim': 'Original data size: 907,740 bytes',
        'theorem': 7,
        'verification': 'Measure Parquet shard sizes',
        'files': ['lmfdb_core_shards/*.parquet'],
        'status': 'unverified'
    },
    {
        'id': 'P10',
        'claim': 'Trainable parameters: 9,690',
        'theorem': 7,
        'verification': 'Count network parameters',
        'files': ['monster_autoencoder.py'],
        'status': 'unverified'
    },
    {
        'id': 'P11',
        'claim': 'Compression ratio: 23×',
        'theorem': 7,
        'verification': 'Calculate 907740 / 38760',
        'files': ['prove_nn_compression.py'],
        'status': 'unverified'
    },
    {
        'id': 'P12',
        'claim': 'Network capacity: 71^5 = 1,804,229,351',
        'theorem': 8,
        'verification': 'Calculate 71^5',
        'files': [],
        'status': 'unverified'
    },
    {
        'id': 'P13',
        'claim': 'Overcapacity: 253,581×',
        'theorem': 8,
        'verification': 'Calculate 1804229351 / 7115',
        'files': ['prove_nn_compression.py'],
        'status': 'unverified'
    },
    {
        'id': 'P14',
        'claim': '71 Hecke operators exist',
        'theorem': 9,
        'verification': 'Count operators in implementation',
        'files': ['monster_autoencoder.py', 'monster_autoencoder_rust.rs'],
        'status': 'unverified'
    },
    {
        'id': 'P15',
        'claim': 'Python and Rust have same architecture',
        'theorem': 10,
        'verification': 'Compare layer definitions',
        'files': ['monster_autoencoder.py', 'monster_autoencoder_rust.rs'],
        'status': 'unverified'
    },
    {
        'id': 'P16',
        'claim': 'Rust MSE: 0.233',
        'theorem': 11,
        'verification': 'Run Rust implementation',
        'files': ['prove_rust_simple.py'],
        'status': 'unverified'
    },
    {
        'id': 'P17',
        'claim': '6 Hecke operators tested',
        'theorem': 12,
        'verification': 'Count test cases',
        'files': ['prove_rust_simple.py', 'monster_autoencoder_rust.rs'],
        'status': 'unverified'
    },
    {
        'id': 'P18',
        'claim': 'Rust execution time: 0.018s (best)',
        'theorem': 13,
        'verification': 'Run benchmark',
        'files': ['prove_rust_simple.py'],
        'status': 'unverified'
    },
    {
        'id': 'P19',
        'claim': '100× speedup estimate',
        'theorem': 13,
        'verification': 'Compare Python vs Rust timing',
        'files': [],
        'status': 'unverified'
    },
    {
        'id': 'P20',
        'claim': 'Rust code compiles without errors',
        'theorem': 14,
        'verification': 'Run cargo check',
        'files': ['lmfdb-rust/'],
        'status': 'unverified'
    },
    {
        'id': 'P21',
        'claim': '3 Rust tests pass',
        'theorem': 15,
        'verification': 'Run cargo test',
        'files': ['monster_autoencoder_rust.rs'],
        'status': 'unverified'
    },
    {
        'id': 'P22',
        'claim': '20 functions converted to Rust',
        'theorem': 16,
        'verification': 'Count converted functions',
        'files': ['lmfdb-rust/src/bin/lmfdb_functions.rs', 'lmfdb_rust_conversion.json'],
        'status': 'unverified'
    },
    {
        'id': 'P23',
        'claim': '500 total Python functions',
        'theorem': 16,
        'verification': 'Count functions in JSON',
        'files': ['lmfdb_math_functions.json'],
        'status': 'unverified'
    },
]

print(f"Total propositions to verify: {len(propositions)}")
print()

for p in propositions:
    print(f"{p['id']}: {p['claim']}")
    print(f"    Theorem: {p['theorem']}")
    print(f"    Method: {p['verification']}")
    print(f"    Files: {', '.join(p['files']) if p['files'] else 'calculation'}")
    print()

# Save propositions
with open('propositions.json', 'w') as f:
    json.dump({
        'theorems': theorems,
        'propositions': propositions,
        'total_theorems': len(theorems),
        'total_propositions': len(propositions),
        'verified': 0,
        'unverified': len(propositions)
    }, f, indent=2)

print("=" * 60)
print("SAVED: propositions.json")
print("=" * 60)
print()

print("Next steps:")
print("1. Run verify_propositions.py to check each claim")
print("2. Use vision model to review paper")
print("3. Generate verification report")
print("4. Update paper with verified claims")
