#!/usr/bin/env python3
"""
Assign complexity levels (1-71) to all LMFDB objects with prime 71
Compute complexity score and topologically sort
"""

import json
import hashlib
from pathlib import Path
from collections import defaultdict

print("🔢 COMPLEXITY SCORING & TOPOLOGICAL SORT")
print("=" * 60)
print()

# Load extracted objects
with open('lmfdb_71_objects.json') as f:
    ast_data = json.load(f)

with open('lmfdb_71_sweep.json') as f:
    pattern_data = json.load(f)

# Define mathematical object types with base complexity
object_types = {
    # Level 1-10: Basic objects
    'prime': {'base_complexity': 1, 'description': 'Prime number 71'},
    'constant': {'base_complexity': 2, 'description': 'Literal constant 71'},
    
    # Level 11-20: Arithmetic objects
    'conductor': {'base_complexity': 11, 'description': 'Conductor = 71'},
    'discriminant': {'base_complexity': 12, 'description': 'Discriminant = 71'},
    'level': {'base_complexity': 13, 'description': 'Level = 71'},
    'degree': {'base_complexity': 14, 'description': 'Degree = 71'},
    
    # Level 21-30: Geometric objects
    'dimension': {'base_complexity': 21, 'description': 'Dimension = 71'},
    'genus': {'base_complexity': 22, 'description': 'Genus = 71'},
    'rank': {'base_complexity': 23, 'description': 'Rank = 71'},
    
    # Level 31-40: Field objects
    'field_size': {'base_complexity': 31, 'description': 'Finite field F_71'},
    'field_extension': {'base_complexity': 32, 'description': 'Field extension degree 71'},
    
    # Level 41-50: Analytic objects
    'coefficient': {'base_complexity': 41, 'description': 'Fourier coefficient a_71'},
    'eigenvalue': {'base_complexity': 42, 'description': 'Hecke eigenvalue λ_71'},
    'hecke': {'base_complexity': 43, 'description': 'Hecke operator T_71'},
    
    # Level 51-60: Modular objects
    'modular_form': {'base_complexity': 51, 'description': 'Modular form with 71'},
    'elliptic_curve': {'base_complexity': 52, 'description': 'Elliptic curve with 71'},
    'abelian_variety': {'base_complexity': 53, 'description': 'Abelian variety over F_71'},
    
    # Level 61-71: Complex objects
    'hypergeometric': {'base_complexity': 61, 'description': 'Hypergeometric motive'},
    'number_field': {'base_complexity': 62, 'description': 'Number field with 71'},
    'galois_group': {'base_complexity': 63, 'description': 'Galois group of order 71'},
}

# Collect all objects
all_objects = []

# From AST parsing
for obj_type, objs in ast_data['objects'].items():
    for obj in objs:
        all_objects.append({
            'source': 'ast',
            'type': obj_type,
            'file': obj['file'],
            'line': obj['line'],
            'value': 71
        })

# From pattern matching
for obj_type, objs in pattern_data['matches'].items():
    for obj in objs:
        all_objects.append({
            'source': 'pattern',
            'type': obj_type,
            'file': obj['file'],
            'line': obj['line'],
            'code': obj.get('code', '')
        })

print(f"Total objects: {len(all_objects)}")
print()

# Compute complexity for each object
scored_objects = []

for obj in all_objects:
    obj_type = obj['type']
    
    # Base complexity from type
    if obj_type in object_types:
        base = object_types[obj_type]['base_complexity']
    else:
        base = 5  # Default for unknown types
    
    # Size complexity: file path length + line number
    file_complexity = len(obj['file']) // 10
    line_complexity = obj['line'] // 100
    
    # Code complexity: length of code
    code_complexity = len(obj.get('code', '')) // 50
    
    # Total complexity
    complexity = base + file_complexity + line_complexity + code_complexity
    
    # Assign level (1-71) using modulo
    level = (complexity % 71) + 1
    
    # Hash for unique ID
    obj_hash = hashlib.sha256(
        f"{obj['file']}:{obj['line']}:{obj['type']}".encode()
    ).hexdigest()[:8]
    
    scored_objects.append({
        'id': obj_hash,
        'type': obj_type,
        'file': obj['file'],
        'line': obj['line'],
        'base_complexity': base,
        'total_complexity': complexity,
        'level': level,
        'code': obj.get('code', '')[:100]
    })

# Sort by complexity
scored_objects.sort(key=lambda x: x['total_complexity'])

print("📊 COMPLEXITY DISTRIBUTION:")
print("-" * 60)

# Count by level
level_counts = defaultdict(int)
for obj in scored_objects:
    level_counts[obj['level']] += 1

print(f"Objects distributed across {len(level_counts)} levels (of 71)")
print()

# Show level distribution
print("Top 10 levels by object count:")
for level, count in sorted(level_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  Level {level:2}: {count:3} objects")

print()

# Topological sort by dependencies
print("🔗 TOPOLOGICAL SORT:")
print("-" * 60)

# Build dependency graph
# Simpler objects depend on more complex ones
dependencies = defaultdict(set)

for i, obj1 in enumerate(scored_objects):
    for j, obj2 in enumerate(scored_objects):
        if i != j:
            # If obj1 is in same file but earlier line, it may depend on obj2
            if obj1['file'] == obj2['file'] and obj1['line'] < obj2['line']:
                if obj1['total_complexity'] < obj2['total_complexity']:
                    dependencies[obj1['id']].add(obj2['id'])

# Topological sort (Kahn's algorithm)
in_degree = defaultdict(int)
for obj_id, deps in dependencies.items():
    for dep in deps:
        in_degree[dep] += 1

queue = [obj['id'] for obj in scored_objects if in_degree[obj['id']] == 0]
sorted_ids = []

while queue:
    obj_id = queue.pop(0)
    sorted_ids.append(obj_id)
    
    for dep in dependencies.get(obj_id, []):
        in_degree[dep] -= 1
        if in_degree[dep] == 0:
            queue.append(dep)

print(f"Topologically sorted {len(sorted_ids)} objects")
print()

# Show sorted objects by level
print("📋 OBJECTS BY LEVEL (TOPOLOGICALLY SORTED):")
print("-" * 60)

# Group by level
by_level = defaultdict(list)
for obj in scored_objects:
    by_level[obj['level']].append(obj)

# Show each level
for level in sorted(by_level.keys())[:10]:  # Show first 10 levels
    objs = by_level[level]
    print(f"\nLevel {level:2} ({len(objs)} objects):")
    for obj in objs[:3]:  # Show first 3 per level
        print(f"  {obj['type']:15} complexity={obj['total_complexity']:3} {obj['file']}:{obj['line']}")
    if len(objs) > 3:
        print(f"  ... and {len(objs) - 3} more")

print()

# Save results
output = {
    'total_objects': len(scored_objects),
    'levels_used': len(level_counts),
    'level_distribution': dict(level_counts),
    'objects': scored_objects,
    'topological_order': sorted_ids,
    'object_types': object_types
}

with open('lmfdb_71_complexity.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"💾 Saved to: lmfdb_71_complexity.json")
print()

# Summary statistics
print("📈 SUMMARY STATISTICS:")
print("-" * 60)
print(f"Total objects: {len(scored_objects)}")
print(f"Levels used: {len(level_counts)}/71")
print(f"Min complexity: {min(obj['total_complexity'] for obj in scored_objects)}")
print(f"Max complexity: {max(obj['total_complexity'] for obj in scored_objects)}")
print(f"Avg complexity: {sum(obj['total_complexity'] for obj in scored_objects) / len(scored_objects):.1f}")
print()

# Most complex objects
print("🏆 TOP 5 MOST COMPLEX OBJECTS:")
print("-" * 60)
for obj in scored_objects[-5:]:
    print(f"Level {obj['level']:2}, Complexity {obj['total_complexity']:3}: {obj['type']}")
    print(f"  {obj['file']}:{obj['line']}")

print()

# Simplest objects
print("🎯 TOP 5 SIMPLEST OBJECTS:")
print("-" * 60)
for obj in scored_objects[:5]:
    print(f"Level {obj['level']:2}, Complexity {obj['total_complexity']:3}: {obj['type']}")
    print(f"  {obj['file']}:{obj['line']}")

print()
print("✅ COMPLEXITY ANALYSIS COMPLETE")
