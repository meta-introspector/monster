#!/usr/bin/env python3
"""
Extract ALL data from LMFDB Python code - no database needed!
Parse Python directly and find discriminant=71, prime 71 patterns
"""

import ast
import re
import json
from pathlib import Path
from collections import defaultdict

LMFDB_PATH = "/mnt/data1/nix/source/github/meta-introspector/lmfdb"
OUTPUT = "lmfdb_extracted_data.json"

print("🔍 EXTRACTING LMFDB DATA FROM PYTHON CODE")
print("=" * 60)
print()

all_data = {
    'discriminant_71': [],
    'prime_71_refs': [],
    'hecke_eigenvalues': [],
    'database_inserts': [],
    'total_numbers': defaultdict(int)
}

py_files = list(Path(LMFDB_PATH).rglob("*.py"))
print(f"Scanning {len(py_files)} Python files...")
print()

for i, py_file in enumerate(py_files):
    if i % 100 == 0:
        print(f"Progress: {i}/{len(py_files)}")
    
    try:
        content = py_file.read_text(encoding='utf-8', errors='ignore')
        
        # Find all numbers
        numbers = re.findall(r'\b\d+\b', content)
        for num in numbers:
            n = int(num)
            if n > 0 and n < 1000000:
                all_data['total_numbers'][n] += 1
        
        # Find discriminant = 71
        if 'discriminant' in content and '71' in content:
            lines = content.split('\n')
            for line_no, line in enumerate(lines, 1):
                if 'discriminant' in line.lower() and '71' in line:
                    all_data['discriminant_71'].append({
                        'file': str(py_file.relative_to(LMFDB_PATH)),
                        'line': line_no,
                        'code': line.strip()
                    })
        
        # Find prime 71 references
        if '71' in content:
            all_data['prime_71_refs'].append({
                'file': str(py_file.relative_to(LMFDB_PATH)),
                'count': content.count('71')
            })
        
        # Find Hecke eigenvalues
        if 'hecke' in content.lower() and 'eigenvalue' in content.lower():
            all_data['hecke_eigenvalues'].append({
                'file': str(py_file.relative_to(LMFDB_PATH)),
                'size': len(content)
            })
        
        # Find database inserts
        if 'db.insert' in content or 'INSERT INTO' in content:
            all_data['database_inserts'].append({
                'file': str(py_file.relative_to(LMFDB_PATH)),
                'has_71': '71' in content
            })
            
    except Exception as e:
        pass

print()
print("✅ EXTRACTION COMPLETE")
print("=" * 60)
print()

# Analyze results
print("📊 RESULTS:")
print()
print(f"Files with discriminant=71: {len(all_data['discriminant_71'])}")
print(f"Files with prime 71: {len([f for f in all_data['prime_71_refs'] if f['count'] > 0])}")
print(f"Files with Hecke eigenvalues: {len(all_data['hecke_eigenvalues'])}")
print(f"Files with database inserts: {len(all_data['database_inserts'])}")
print()

# Top numbers
print("Top 20 numbers in LMFDB code:")
top_numbers = sorted(all_data['total_numbers'].items(), key=lambda x: -x[1])[:20]
for num, count in top_numbers:
    div_71 = "✓" if num % 71 == 0 else ""
    print(f"  {num:6}: {count:5} occurrences {div_71}")

print()
print(f"Number 71 appears: {all_data['total_numbers'][71]} times")
print()

# Show discriminant=71 cases
if all_data['discriminant_71']:
    print("🎯 DISCRIMINANT=71 CASES:")
    for case in all_data['discriminant_71'][:10]:
        print(f"  {case['file']}:{case['line']}")
        print(f"    {case['code'][:80]}")

# Save
with open(OUTPUT, 'w') as f:
    json.dump(all_data, f, indent=2, default=int)

print()
print(f"💾 Saved to: {OUTPUT}")
print()
print(f"Total unique numbers found: {len(all_data['total_numbers'])}")
