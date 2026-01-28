#!/usr/bin/env python3
"""
Sweep ALL LMFDB mathematical objects for prime 71
- Conductors
- Discriminants
- Levels
- Degrees
- Dimensions
- Field sizes
- Coefficients
"""

import json
import re
from pathlib import Path
from collections import defaultdict

LMFDB_PATH = "/mnt/data1/nix/source/github/meta-introspector/lmfdb"

print("🔍 SWEEPING ALL LMFDB OBJECTS FOR PRIME 71")
print("=" * 60)
print()

# Mathematical object types to search for
object_types = {
    'conductor': r'conductor.*=.*71|conductor.*:.*71',
    'discriminant': r'discriminant.*=.*71|discriminant.*:.*71',
    'level': r'level.*=.*71|level.*:.*71',
    'degree': r'degree.*=.*71|degree.*:.*71',
    'dimension': r'dimension.*=.*71|dimension.*:.*71',
    'field_size': r'field.*=.*71|GF\(71\)|FiniteField\(71\)',
    'prime': r'prime.*=.*71|p.*=.*71',
    'order': r'order.*=.*71',
    'rank': r'rank.*=.*71',
    'genus': r'genus.*=.*71',
    'coefficient': r'a_71|coeff.*71',
    'eigenvalue': r'eigenvalue.*71|lambda.*71',
    'hecke': r'hecke.*71|T_71',
}

results = defaultdict(list)
total_matches = 0

py_files = list(Path(LMFDB_PATH).rglob("*.py"))
print(f"Scanning {len(py_files)} Python files...")
print()

for py_file in py_files:
    try:
        content = py_file.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')
        
        for line_no, line in enumerate(lines, 1):
            if '71' not in line:
                continue
                
            # Check each object type
            for obj_type, pattern in object_types.items():
                if re.search(pattern, line, re.IGNORECASE):
                    results[obj_type].append({
                        'file': str(py_file.relative_to(LMFDB_PATH)),
                        'line': line_no,
                        'code': line.strip()[:200]  # Truncate long lines
                    })
                    total_matches += 1
                    
    except Exception as e:
        pass

print("📊 RESULTS BY OBJECT TYPE:")
print("-" * 60)

for obj_type in sorted(object_types.keys()):
    count = len(results[obj_type])
    if count > 0:
        print(f"{obj_type:15}: {count:4} matches")

print()
print(f"Total matches: {total_matches}")
print()

# Show top matches for each type
print("🎯 TOP MATCHES BY TYPE:")
print("-" * 60)

for obj_type in sorted(object_types.keys()):
    matches = results[obj_type]
    if matches:
        print(f"\n{obj_type.upper()}:")
        for match in matches[:3]:
            print(f"  {match['file']}:{match['line']}")
            print(f"    {match['code']}")

# Save results
output = {
    'total_matches': total_matches,
    'by_type': {k: len(v) for k, v in results.items()},
    'matches': {k: v for k, v in results.items()}
}

with open('lmfdb_71_sweep.json', 'w') as f:
    json.dump(output, f, indent=2)

print()
print(f"💾 Saved to: lmfdb_71_sweep.json")
print()

# Find files with most 71 references
print("📁 FILES WITH MOST PRIME 71 REFERENCES:")
print("-" * 60)

file_counts = defaultdict(int)
for obj_type, matches in results.items():
    for match in matches:
        file_counts[match['file']] += 1

top_files = sorted(file_counts.items(), key=lambda x: -x[1])[:10]
for file, count in top_files:
    print(f"{count:3} matches: {file}")

print()
print("✅ SWEEP COMPLETE")
