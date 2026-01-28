#!/usr/bin/env python3
"""
Extract ALL mathematical objects with value 71 from LMFDB
Parse actual data structures, not just grep
"""

import ast
import json
from pathlib import Path
from collections import defaultdict

LMFDB_PATH = "/mnt/data1/nix/source/github/meta-introspector/lmfdb"

print("🔬 EXTRACTING ALL 71-VALUED OBJECTS FROM LMFDB")
print("=" * 60)
print()

objects_found = defaultdict(list)
total_objects = 0

py_files = list(Path(LMFDB_PATH).rglob("*.py"))
print(f"Parsing {len(py_files)} Python files...")
print()

for i, py_file in enumerate(py_files):
    if i % 100 == 0:
        print(f"Progress: {i}/{len(py_files)}")
    
    try:
        content = py_file.read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(content)
        
        # Walk AST looking for 71
        for node in ast.walk(tree):
            # Assignments: x = 71
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Constant) and node.value.value == 71:
                            objects_found['assignment'].append({
                                'file': str(py_file.relative_to(LMFDB_PATH)),
                                'line': node.lineno,
                                'name': target.id,
                                'value': 71,
                                'type': 'assignment'
                            })
                            total_objects += 1
            
            # Function calls: func(71) or func(x=71)
            elif isinstance(node, ast.Call):
                # Positional args
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value == 71:
                        func_name = ast.unparse(node.func) if hasattr(ast, 'unparse') else 'unknown'
                        objects_found['function_arg'].append({
                            'file': str(py_file.relative_to(LMFDB_PATH)),
                            'line': node.lineno,
                            'function': func_name,
                            'value': 71,
                            'type': 'function_arg'
                        })
                        total_objects += 1
                
                # Keyword args
                for keyword in node.keywords:
                    if isinstance(keyword.value, ast.Constant) and keyword.value.value == 71:
                        func_name = ast.unparse(node.func) if hasattr(ast, 'unparse') else 'unknown'
                        objects_found['keyword_arg'].append({
                            'file': str(py_file.relative_to(LMFDB_PATH)),
                            'line': node.lineno,
                            'function': func_name,
                            'keyword': keyword.arg,
                            'value': 71,
                            'type': 'keyword_arg'
                        })
                        total_objects += 1
            
            # Comparisons: x == 71, x > 71, etc.
            elif isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and comparator.value == 71:
                        objects_found['comparison'].append({
                            'file': str(py_file.relative_to(LMFDB_PATH)),
                            'line': node.lineno,
                            'value': 71,
                            'type': 'comparison'
                        })
                        total_objects += 1
            
            # List/tuple literals: [71, ...]
            elif isinstance(node, (ast.List, ast.Tuple)):
                for elt in node.elts:
                    if isinstance(elt, ast.Constant) and elt.value == 71:
                        objects_found['collection'].append({
                            'file': str(py_file.relative_to(LMFDB_PATH)),
                            'line': node.lineno,
                            'value': 71,
                            'type': 'collection'
                        })
                        total_objects += 1
            
            # Dict literals: {x: 71} or {71: x}
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if key and isinstance(key, ast.Constant) and key.value == 71:
                        objects_found['dict_key'].append({
                            'file': str(py_file.relative_to(LMFDB_PATH)),
                            'line': node.lineno,
                            'value': 71,
                            'type': 'dict_key'
                        })
                        total_objects += 1
                    if isinstance(value, ast.Constant) and value.value == 71:
                        objects_found['dict_value'].append({
                            'file': str(py_file.relative_to(LMFDB_PATH)),
                            'line': node.lineno,
                            'value': 71,
                            'type': 'dict_value'
                        })
                        total_objects += 1
                        
    except Exception as e:
        pass

print()
print("📊 OBJECTS FOUND BY TYPE:")
print("-" * 60)

for obj_type in sorted(objects_found.keys()):
    count = len(objects_found[obj_type])
    print(f"{obj_type:15}: {count:4} objects")

print()
print(f"Total objects: {total_objects}")
print()

# Show examples
print("🎯 EXAMPLES BY TYPE:")
print("-" * 60)

for obj_type in sorted(objects_found.keys()):
    objs = objects_found[obj_type]
    if objs:
        print(f"\n{obj_type.upper()}:")
        for obj in objs[:3]:
            print(f"  {obj['file']}:{obj['line']}")
            if 'name' in obj:
                print(f"    {obj['name']} = 71")
            elif 'function' in obj:
                if 'keyword' in obj:
                    print(f"    {obj['function']}({obj['keyword']}=71)")
                else:
                    print(f"    {obj['function']}(71)")

# Save
output = {
    'total_objects': total_objects,
    'by_type': {k: len(v) for k, v in objects_found.items()},
    'objects': {k: v for k, v in objects_found.items()}
}

with open('lmfdb_71_objects.json', 'w') as f:
    json.dump(output, f, indent=2)

print()
print(f"💾 Saved to: lmfdb_71_objects.json")
print()

# Categorize by mathematical meaning
print("🔮 CATEGORIZATION BY MATHEMATICAL MEANING:")
print("-" * 60)

# Try to infer meaning from variable names
categories = defaultdict(int)

for obj_type, objs in objects_found.items():
    for obj in objs:
        name = obj.get('name', obj.get('keyword', obj.get('function', ''))).lower()
        
        if any(x in name for x in ['conductor', 'cond']):
            categories['conductor'] += 1
        elif any(x in name for x in ['discriminant', 'disc']):
            categories['discriminant'] += 1
        elif any(x in name for x in ['level', 'lev']):
            categories['level'] += 1
        elif any(x in name for x in ['field', 'gf', 'finite']):
            categories['field_size'] += 1
        elif any(x in name for x in ['prime', 'p']):
            categories['prime'] += 1
        elif any(x in name for x in ['dimension', 'dim']):
            categories['dimension'] += 1
        elif any(x in name for x in ['degree', 'deg']):
            categories['degree'] += 1
        else:
            categories['other'] += 1

for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"{cat:15}: {count:4} objects")

print()
print("✅ EXTRACTION COMPLETE")
