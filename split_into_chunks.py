#!/usr/bin/env python3
"""
Split 40 files with prime 71 into smallest chunks:
- Functions
- Classes  
- Statements
- Expressions
Then measure each chunk
"""

import ast
import json
from pathlib import Path

LMFDB_PATH = "/mnt/data1/nix/source/github/meta-introspector/lmfdb"

# Load files with 71
with open('lmfdb_extracted_data.json') as f:
    data = json.load(f)

files_with_71 = [f['file'] for f in data['prime_71_refs'] if f['count'] > 0]

print(f"🔬 SPLITTING {len(files_with_71)} FILES INTO CHUNKS")
print("=" * 60)
print()

all_chunks = []
total_chunks = 0

for file_path in files_with_71:
    full_path = Path(LMFDB_PATH) / file_path
    
    try:
        content = full_path.read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(content)
        
        # Extract functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno + 1
                func_code = '\n'.join(content.split('\n')[node.lineno-1:node.end_lineno])
                has_71 = '71' in func_code
                
                chunk = {
                    'type': 'function',
                    'file': file_path,
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno,
                    'lines': func_lines,
                    'bytes': len(func_code),
                    'has_71': has_71,
                    'code': func_code if has_71 else None
                }
                all_chunks.append(chunk)
                total_chunks += 1
                
            elif isinstance(node, ast.ClassDef):
                class_lines = node.end_lineno - node.lineno + 1
                class_code = '\n'.join(content.split('\n')[node.lineno-1:node.end_lineno])
                has_71 = '71' in class_code
                
                chunk = {
                    'type': 'class',
                    'file': file_path,
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno,
                    'lines': class_lines,
                    'bytes': len(class_code),
                    'has_71': has_71,
                    'code': class_code if has_71 else None
                }
                all_chunks.append(chunk)
                total_chunks += 1
                
    except Exception as e:
        print(f"✗ {file_path}: {e}")

print(f"✅ Extracted {total_chunks} chunks")
print()

# Filter chunks with 71
chunks_with_71 = [c for c in all_chunks if c['has_71']]
print(f"🎯 Chunks containing 71: {len(chunks_with_71)}")
print()

# Sort by size
chunks_with_71.sort(key=lambda x: x['bytes'])

print("📊 SMALLEST CHUNKS WITH 71:")
print("-" * 60)
for chunk in chunks_with_71[:10]:
    print(f"{chunk['type']:8} {chunk['name']:30} {chunk['lines']:3} lines, {chunk['bytes']:5} bytes")
    print(f"  {chunk['file']}")

print()
print("📊 LARGEST CHUNKS WITH 71:")
print("-" * 60)
for chunk in chunks_with_71[-10:]:
    print(f"{chunk['type']:8} {chunk['name']:30} {chunk['lines']:3} lines, {chunk['bytes']:5} bytes")
    print(f"  {chunk['file']}")

# Save chunks
with open('lmfdb_71_chunks.json', 'w') as f:
    json.dump(chunks_with_71, f, indent=2)

print()
print(f"💾 Saved {len(chunks_with_71)} chunks to: lmfdb_71_chunks.json")
print()

# Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total chunks: {total_chunks}")
print(f"Chunks with 71: {len(chunks_with_71)}")
print(f"Smallest: {chunks_with_71[0]['bytes']} bytes")
print(f"Largest: {chunks_with_71[-1]['bytes']} bytes")
print(f"Average: {sum(c['bytes'] for c in chunks_with_71) // len(chunks_with_71)} bytes")
