#!/usr/bin/env python3
"""
Expand analysis to ALL 356 files to get 3M+ rows
"""

import json
import ast
from pathlib import Path
from collections import defaultdict

LMFDB_PATH = "/mnt/data1/nix/source/github/meta-introspector/lmfdb"
OUTPUT_DIR = "lmfdb_hecke_analysis"
MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

def hecke_resonance(value):
    """Find Monster prime with highest divisibility"""
    if value == 0:
        return 1
    resonances = {}
    for p in MONSTER_PRIMES:
        if value % p == 0:
            count = 0
            temp = abs(value)
            while temp % p == 0:
                count += 1
                temp //= p
            resonances[p] = count
    return max(resonances.items(), key=lambda x: x[1])[0] if resonances else 1

def analyze_all_files():
    """Analyze ALL 356 Python files"""
    
    py_files = list(Path(LMFDB_PATH).rglob("*.py"))
    print(f"🔍 Analyzing {len(py_files)} Python files...")
    print()
    
    # Data structures for 3M+ rows
    all_rows = []
    
    for i, py_file in enumerate(py_files):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(py_files)} files ({len(all_rows):,} rows)")
        
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            rel_path = str(py_file.relative_to(LMFDB_PATH))
            
            # Row 1: File metadata
            all_rows.append({
                'type': 'file',
                'path': rel_path,
                'lines': len(content.split('\n')),
                'bytes': len(content),
                'literal_71': content.count('71'),
                'shard': hecke_resonance(len(content))
            })
            
            # Rows 2-N: Each line
            for line_no, line in enumerate(content.split('\n'), 1):
                if line.strip():  # Non-empty lines
                    all_rows.append({
                        'type': 'line',
                        'path': rel_path,
                        'line_no': line_no,
                        'length': len(line),
                        'has_71': '71' in line,
                        'shard': hecke_resonance(line_no)
                    })
            
            # Rows N+1-M: AST nodes
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    all_rows.append({
                        'type': 'ast_node',
                        'path': rel_path,
                        'node_type': node.__class__.__name__,
                        'line': getattr(node, 'lineno', 0),
                        'is_constant_71': isinstance(node, ast.Constant) and node.value == 71,
                        'shard': hecke_resonance(getattr(node, 'lineno', 1))
                    })
            except:
                pass
                
        except Exception as e:
            pass
    
    print(f"\n✅ Complete: {len(all_rows):,} total rows")
    
    # Save in chunks
    chunk_size = 100000
    for i in range(0, len(all_rows), chunk_size):
        chunk = all_rows[i:i+chunk_size]
        chunk_file = f"{OUTPUT_DIR}/full_analysis_chunk_{i//chunk_size:03d}.json"
        with open(chunk_file, 'w') as f:
            json.dump(chunk, f)
        print(f"  Saved: {chunk_file} ({len(chunk):,} rows)")
    
    # Summary
    summary = {
        'total_rows': len(all_rows),
        'files': len(py_files),
        'by_type': defaultdict(int),
        'by_shard': defaultdict(int),
        'with_71': sum(1 for r in all_rows if r.get('has_71') or r.get('is_constant_71') or r.get('literal_71', 0) > 0)
    }
    
    for row in all_rows:
        summary['by_type'][row['type']] += 1
        summary['by_shard'][row.get('shard', 1)] += 1
    
    with open(f"{OUTPUT_DIR}/full_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=int)
    
    print(f"\n📊 SUMMARY")
    print(f"Total rows: {summary['total_rows']:,}")
    print(f"Files: {summary['files']}")
    print(f"Rows with 71: {summary['with_71']:,}")
    print(f"\nBy type:")
    for typ, count in sorted(summary['by_type'].items(), key=lambda x: -x[1]):
        print(f"  {typ}: {count:,}")
    print(f"\nBy shard (top 10):")
    for shard, count in sorted(summary['by_shard'].items(), key=lambda x: -x[1])[:10]:
        print(f"  Shard {shard}: {count:,}")

if __name__ == '__main__':
    analyze_all_files()
