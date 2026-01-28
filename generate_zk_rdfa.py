#!/usr/bin/env python3
"""
Generate ZK-RDFa ontology for LMFDB with Monster group symmetry
Compressed form with ontological mappings
"""

import json
import hashlib
from pathlib import Path

print("🔮 GENERATING ZK-RDFA ONTOLOGY WITH MONSTER SYMMETRY")
print("=" * 60)
print()

# Load data
with open('lmfdb_71_complexity.json') as f:
    complexity = json.load(f)

objects = complexity['objects']

# Define Monster-symmetric ontology
ontology = {
    '@context': {
        '@vocab': 'http://lmfdb.org/ontology/',
        'monster': 'http://lmfdb.org/monster/',
        'zk': 'http://lmfdb.org/zk/',
        'hecke': 'http://lmfdb.org/hecke/',
        'shard': 'monster:shard',
        'level': 'monster:level',
        'eigenvalue': 'hecke:eigenvalue',
        'witness': 'zk:witness',
        'proof': 'zk:proof',
        'complexity': 'monster:complexity',
        'prime71': 'monster:prime71'
    },
    '@graph': []
}

print("📊 CREATING ONTOLOGICAL MAPPINGS:")
print("-" * 60)

# Map each object to RDFa
for obj in objects:
    # Compute Monster-symmetric properties
    shard_id = int(hashlib.sha256(obj['id'].encode()).hexdigest(), 16) % 71
    chunk_id = int(hashlib.sha256(obj['file'].encode()).hexdigest(), 16) % 71
    witness_id = obj['line'] % 71
    
    # Hecke eigenvalue (T_71 operator)
    eigenvalue = (obj['total_complexity'] + obj['level']) % 71
    
    # ZK proof
    zk_proof = {
        'commitment': hashlib.sha256(f"{obj['id']}:{obj['line']}".encode()).hexdigest()[:32],
        'challenge': (obj['total_complexity'] * 71) % (2**256),
        'response': (obj['level'] * eigenvalue) % 71
    }
    
    # RDFa node
    node = {
        '@id': f"monster:object/{obj['id']}",
        '@type': f"monster:{obj['type']}",
        'shard': shard_id,
        'chunk': chunk_id,
        'witness': witness_id,
        'level': obj['level'],
        'complexity': obj['total_complexity'],
        'eigenvalue': eigenvalue,
        'file': obj['file'],
        'line': obj['line'],
        'zk:proof': {
            '@type': 'zk:SchnorrProof',
            'commitment': zk_proof['commitment'],
            'challenge': str(zk_proof['challenge']),
            'response': zk_proof['response']
        },
        'monster:factorization': {
            'shard_mod_71': shard_id,
            'chunk_mod_71': chunk_id,
            'witness_mod_71': witness_id,
            'eigenvalue_mod_71': eigenvalue
        }
    }
    
    ontology['@graph'].append(node)

print(f"Created {len(ontology['@graph'])} RDFa nodes")
print()

# Add Monster group structure
print("🔮 ADDING MONSTER GROUP STRUCTURE:")
print("-" * 60)

monster_structure = {
    '@id': 'monster:group',
    '@type': 'monster:SporadicGroup',
    'name': 'Monster Group',
    'order': '808017424794512875886459904961710757005754368000000000',
    'primes': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71],
    'prime71': {
        '@id': 'monster:prime71',
        '@type': 'monster:LargestPrime',
        'value': 71,
        'exponent': 1,
        'shards': len(set(n['shard'] for n in ontology['@graph'])),
        'resonance': 'dominant'
    },
    'hecke_operators': {
        'T_71': {
            '@type': 'hecke:Operator',
            'prime': 71,
            'eigenvalues': list(set(n['eigenvalue'] for n in ontology['@graph']))
        }
    }
}

ontology['@graph'].append(monster_structure)

print("✅ Added Monster group structure")
print()

# Add shard ontology
print("📦 ADDING SHARD ONTOLOGY:")
print("-" * 60)

shards_by_id = {}
for node in ontology['@graph']:
    if '@type' in node and node['@type'] != 'monster:SporadicGroup':
        shard_id = node['shard']
        if shard_id not in shards_by_id:
            shards_by_id[shard_id] = []
        shards_by_id[shard_id].append(node['@id'])

for shard_id, members in shards_by_id.items():
    shard_node = {
        '@id': f'monster:shard/{shard_id}',
        '@type': 'monster:Shard',
        'shard_id': shard_id,
        'member_count': len(members),
        'members': members,
        'mod_71': shard_id,
        'is_monster_prime': shard_id in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]
    }
    ontology['@graph'].append(shard_node)

print(f"Created {len(shards_by_id)} shard nodes")
print()

# Save full ontology
with open('lmfdb_monster_ontology.jsonld', 'w') as f:
    json.dump(ontology, f, indent=2)

print(f"💾 Saved full ontology: lmfdb_monster_ontology.jsonld")
print()

# Create compressed form
print("🗜️  CREATING COMPRESSED ZK-RDFA:")
print("-" * 60)

compressed = {
    '@context': ontology['@context'],
    'version': '1.0',
    'compression': 'monster71',
    'objects': []
}

for node in ontology['@graph']:
    if '@type' in node and node['@type'] != 'monster:SporadicGroup' and node['@type'] != 'monster:Shard':
        # Compress to minimal form
        compressed_node = {
            'id': node['@id'].split('/')[-1],
            't': node['@type'].split(':')[-1],
            's': node['shard'],
            'c': node['chunk'],
            'w': node['witness'],
            'l': node['level'],
            'e': node['eigenvalue'],
            'p': node['zk:proof']['commitment'][:16]  # Truncate proof
        }
        compressed['objects'].append(compressed_node)

# Add compression metadata
compressed['meta'] = {
    'total': len(compressed['objects']),
    'shards': len(shards_by_id),
    'compression_ratio': len(json.dumps(compressed)) / len(json.dumps(ontology))
}

with open('lmfdb_monster_compressed.jsonld', 'w') as f:
    json.dump(compressed, f, indent=2)

print(f"💾 Saved compressed: lmfdb_monster_compressed.jsonld")
print(f"Compression ratio: {compressed['meta']['compression_ratio']:.2%}")
print()

# Generate Turtle format
print("🐢 GENERATING TURTLE (TTL) FORMAT:")
print("-" * 60)

ttl = """@prefix monster: <http://lmfdb.org/monster/> .
@prefix zk: <http://lmfdb.org/zk/> .
@prefix hecke: <http://lmfdb.org/hecke/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

# Monster Group
monster:group a monster:SporadicGroup ;
    rdfs:label "Monster Group" ;
    monster:order "808017424794512875886459904961710757005754368000000000" ;
    monster:largestPrime monster:prime71 .

monster:prime71 a monster:LargestPrime ;
    monster:value 71 ;
    monster:exponent 1 ;
    monster:shards %d .

""" % len(shards_by_id)

# Add objects
for node in ontology['@graph']:
    if '@type' in node and node['@type'] not in ['monster:SporadicGroup', 'monster:Shard']:
        ttl += f"""
monster:object/{node['@id'].split('/')[-1]} a monster:{node['@type'].split(':')[-1]} ;
    monster:shard {node['shard']} ;
    monster:level {node['level']} ;
    hecke:eigenvalue {node['eigenvalue']} ;
    zk:proof "{node['zk:proof']['commitment'][:16]}" .
"""

with open('lmfdb_monster_ontology.ttl', 'w') as f:
    f.write(ttl)

print(f"💾 Saved Turtle: lmfdb_monster_ontology.ttl")
print()

# Statistics
print("=" * 60)
print("ONTOLOGY STATISTICS")
print("=" * 60)
print()

print(f"Total nodes: {len(ontology['@graph'])}")
print(f"Object nodes: {len([n for n in ontology['@graph'] if '@type' in n and 'object' in n.get('@id', '')])}")
print(f"Shard nodes: {len(shards_by_id)}")
print(f"Unique eigenvalues: {len(set(n['eigenvalue'] for n in ontology['@graph'] if 'eigenvalue' in n))}")
print()

print("File sizes:")
full_size = Path('lmfdb_monster_ontology.jsonld').stat().st_size
compressed_size = Path('lmfdb_monster_compressed.jsonld').stat().st_size
ttl_size = Path('lmfdb_monster_ontology.ttl').stat().st_size

print(f"  Full JSON-LD: {full_size:,} bytes")
print(f"  Compressed:   {compressed_size:,} bytes ({100*compressed_size/full_size:.1f}%)")
print(f"  Turtle:       {ttl_size:,} bytes ({100*ttl_size/full_size:.1f}%)")
print()

print("✅ ZK-RDFA ONTOLOGY COMPLETE")
