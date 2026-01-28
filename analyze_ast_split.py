#!/usr/bin/env python3
"""
Split Python code into constants and functions
Rank by complexity level (1-71)
"""

import ast
import json
from pathlib import Path
from collections import defaultdict

print("🔬 ANALYZING PYTHON AST: CONSTANTS vs FUNCTIONS")
print("=" * 60)
print()

# Load complexity data
with open('lmfdb_71_complexity.json') as f:
    complexity_data = json.load(f)

LMFDB_PATH = "/mnt/data1/nix/source/github/meta-introspector/lmfdb"

# Collect all AST nodes
constants = []
functions = []
classes = []
assignments = []

print("Parsing Python files...")
py_files = list(Path(LMFDB_PATH).rglob("*.py"))

for i, py_file in enumerate(py_files):
    if i % 100 == 0:
        print(f"Progress: {i}/{len(py_files)}")
    
    try:
        content = py_file.read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            # Constants
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float, str, bool)):
                    # Compute complexity
                    if isinstance(node.value, int):
                        complexity = abs(node.value) % 71 if node.value != 0 else 1
                    elif isinstance(node.value, str):
                        complexity = len(node.value) % 71 if node.value else 1
                    else:
                        complexity = 1
                    
                    level = (complexity % 71) + 1
                    
                    constants.append({
                        'type': 'constant',
                        'value': str(node.value)[:100],
                        'value_type': type(node.value).__name__,
                        'file': str(py_file.relative_to(LMFDB_PATH)),
                        'line': node.lineno,
                        'complexity': complexity,
                        'level': level,
                        'has_71': '71' in str(node.value)
                    })
            
            # Functions
            elif isinstance(node, ast.FunctionDef):
                # Count statements
                num_stmts = len(node.body)
                num_args = len(node.args.args)
                
                # Complexity = statements + args
                complexity = num_stmts + num_args
                level = (complexity % 71) + 1
                
                functions.append({
                    'type': 'function',
                    'name': node.name,
                    'file': str(py_file.relative_to(LMFDB_PATH)),
                    'line': node.lineno,
                    'num_statements': num_stmts,
                    'num_args': num_args,
                    'complexity': complexity,
                    'level': level,
                    'has_71': '71' in ast.unparse(node) if hasattr(ast, 'unparse') else False
                })
            
            # Classes
            elif isinstance(node, ast.ClassDef):
                # Count methods
                num_methods = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                
                complexity = num_methods
                level = (complexity % 71) + 1
                
                classes.append({
                    'type': 'class',
                    'name': node.name,
                    'file': str(py_file.relative_to(LMFDB_PATH)),
                    'line': node.lineno,
                    'num_methods': num_methods,
                    'complexity': complexity,
                    'level': level,
                    'has_71': '71' in ast.unparse(node) if hasattr(ast, 'unparse') else False
                })
            
            # Assignments
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check if RHS is constant
                        is_constant = isinstance(node.value, ast.Constant)
                        
                        complexity = 1
                        if is_constant and isinstance(node.value.value, int):
                            complexity = abs(node.value.value) % 71 if node.value.value != 0 else 1
                        
                        level = (complexity % 71) + 1
                        
                        assignments.append({
                            'type': 'assignment',
                            'name': target.id,
                            'is_constant': is_constant,
                            'file': str(py_file.relative_to(LMFDB_PATH)),
                            'line': node.lineno,
                            'complexity': complexity,
                            'level': level,
                            'has_71': '71' in ast.unparse(node) if hasattr(ast, 'unparse') else False
                        })
                        
    except Exception as e:
        pass

print()
print("📊 ANALYSIS COMPLETE")
print("=" * 60)
print()

print(f"Constants: {len(constants)}")
print(f"Functions: {len(functions)}")
print(f"Classes: {len(classes)}")
print(f"Assignments: {len(assignments)}")
print()

# Rank by complexity
print("🏆 TOP 10 MOST COMPLEX FUNCTIONS:")
print("-" * 60)
functions_sorted = sorted(functions, key=lambda x: -x['complexity'])
for func in functions_sorted[:10]:
    print(f"Level {func['level']:2}, Complexity {func['complexity']:3}: {func['name']}")
    print(f"  {func['file']}:{func['line']}")

print()
print("🎯 TOP 10 SIMPLEST FUNCTIONS:")
print("-" * 60)
for func in functions_sorted[-10:]:
    print(f"Level {func['level']:2}, Complexity {func['complexity']:3}: {func['name']}")
    print(f"  {func['file']}:{func['line']}")

print()

# Constants with 71
constants_71 = [c for c in constants if c['has_71']]
print(f"📊 CONSTANTS WITH 71: {len(constants_71)}")
print("-" * 60)
for const in constants_71[:10]:
    print(f"Level {const['level']:2}: {const['value'][:50]}")
    print(f"  {const['file']}:{const['line']}")

print()

# Functions with 71
functions_71 = [f for f in functions if f['has_71']]
print(f"📊 FUNCTIONS WITH 71: {len(functions_71)}")
print("-" * 60)
for func in functions_71[:10]:
    print(f"Level {func['level']:2}, Complexity {func['complexity']:3}: {func['name']}")
    print(f"  {func['file']}:{func['line']}")

print()

# Level distribution
print("📊 LEVEL DISTRIBUTION:")
print("-" * 60)

level_counts = defaultdict(lambda: {'constants': 0, 'functions': 0, 'classes': 0, 'assignments': 0})

for const in constants:
    level_counts[const['level']]['constants'] += 1

for func in functions:
    level_counts[func['level']]['functions'] += 1

for cls in classes:
    level_counts[cls['level']]['classes'] += 1

for assign in assignments:
    level_counts[assign['level']]['assignments'] += 1

# Show top 10 levels
top_levels = sorted(level_counts.items(), 
                   key=lambda x: sum(x[1].values()), 
                   reverse=True)[:10]

for level, counts in top_levels:
    total = sum(counts.values())
    print(f"Level {level:2}: {total:6} total "
          f"(C:{counts['constants']:5}, F:{counts['functions']:4}, "
          f"Cl:{counts['classes']:3}, A:{counts['assignments']:5})")

print()

# Save results
output = {
    'constants': constants[:1000],  # Limit to 1000
    'functions': functions,
    'classes': classes,
    'assignments': assignments[:1000],
    'stats': {
        'total_constants': len(constants),
        'total_functions': len(functions),
        'total_classes': len(classes),
        'total_assignments': len(assignments),
        'constants_with_71': len(constants_71),
        'functions_with_71': len(functions_71)
    },
    'level_distribution': {str(k): v for k, v in level_counts.items()}
}

with open('lmfdb_ast_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"💾 Saved to: lmfdb_ast_analysis.json")
print()

# Summary by type
print("=" * 60)
print("SUMMARY BY TYPE")
print("=" * 60)
print()

print("CONSTANTS:")
print(f"  Total: {len(constants):,}")
print(f"  With 71: {len(constants_71)}")
print(f"  Levels used: {len(set(c['level'] for c in constants))}/71")
print()

print("FUNCTIONS:")
print(f"  Total: {len(functions):,}")
print(f"  With 71: {len(functions_71)}")
print(f"  Levels used: {len(set(f['level'] for f in functions))}/71")
print(f"  Avg complexity: {sum(f['complexity'] for f in functions) / len(functions):.1f}")
print()

print("CLASSES:")
print(f"  Total: {len(classes):,}")
print(f"  Levels used: {len(set(c['level'] for c in classes))}/71")
print(f"  Avg methods: {sum(c['num_methods'] for c in classes) / len(classes):.1f}")
print()

print("✅ ANALYSIS COMPLETE")
