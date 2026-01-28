#!/usr/bin/env python3
"""
Extract pure mathematical functions from LMFDB
Strip system code, focus on mathematical operations
Model each function mathematically
"""

import ast
import json
from pathlib import Path
from collections import defaultdict

print("🔬 EXTRACTING PURE MATHEMATICAL FUNCTIONS")
print("=" * 60)
print()

LMFDB_PATH = "/mnt/data1/nix/source/github/meta-introspector/lmfdb"

# Mathematical operations to detect
MATH_OPS = {
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift
}

MATH_FUNCTIONS = {
    'gcd', 'lcm', 'sqrt', 'pow', 'abs', 'sum', 'min', 'max',
    'factorial', 'prime', 'divisor', 'factor', 'mod', 'inverse',
    'eigenvalue', 'determinant', 'trace', 'rank', 'dimension',
    'conductor', 'discriminant', 'level', 'degree', 'genus',
    'hecke', 'fourier', 'coefficient', 'expansion'
}

def is_math_function(node):
    """Check if function is mathematical"""
    if not isinstance(node, ast.FunctionDef):
        return False
    
    # Check function name
    name_lower = node.name.lower()
    if any(math_word in name_lower for math_word in MATH_FUNCTIONS):
        return True
    
    # Check for mathematical operations in body
    math_op_count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and type(child.op) in MATH_OPS:
            math_op_count += 1
        elif isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                if child.func.id in MATH_FUNCTIONS:
                    math_op_count += 1
    
    return math_op_count >= 2  # At least 2 math operations

def extract_math_operations(node):
    """Extract mathematical operations from function"""
    operations = []
    
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp):
            op_name = type(child.op).__name__
            operations.append({
                'type': 'binop',
                'operation': op_name,
                'line': child.lineno
            })
        elif isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                if child.func.id in MATH_FUNCTIONS:
                    operations.append({
                        'type': 'call',
                        'function': child.func.id,
                        'line': child.lineno
                    })
    
    return operations

def compute_cyclomatic_complexity(node):
    """Compute cyclomatic complexity"""
    complexity = 1  # Base complexity
    
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    
    return complexity

# Collect mathematical functions
math_functions = []

print("Scanning Python files for mathematical functions...")
py_files = list(Path(LMFDB_PATH).rglob("*.py"))

for i, py_file in enumerate(py_files):
    if i % 100 == 0:
        print(f"Progress: {i}/{len(py_files)}")
    
    try:
        content = py_file.read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if is_math_function(node):
                    # Extract mathematical properties
                    operations = extract_math_operations(node)
                    cyclomatic = compute_cyclomatic_complexity(node)
                    
                    # Count different types of operations
                    num_binops = len([op for op in operations if op['type'] == 'binop'])
                    num_calls = len([op for op in operations if op['type'] == 'call'])
                    
                    # Mathematical complexity
                    math_complexity = num_binops + num_calls * 2 + cyclomatic
                    level = (math_complexity % 71) + 1
                    
                    # Extract function signature
                    args = [arg.arg for arg in node.args.args]
                    
                    math_functions.append({
                        'name': node.name,
                        'file': str(py_file.relative_to(LMFDB_PATH)),
                        'line': node.lineno,
                        'args': args,
                        'num_args': len(args),
                        'operations': operations,
                        'num_binops': num_binops,
                        'num_calls': num_calls,
                        'cyclomatic_complexity': cyclomatic,
                        'math_complexity': math_complexity,
                        'level': level,
                        'has_71': '71' in ast.unparse(node) if hasattr(ast, 'unparse') else False
                    })
                    
    except Exception as e:
        pass

print()
print(f"✅ Found {len(math_functions)} mathematical functions")
print()

# Sort by complexity
math_functions.sort(key=lambda x: -x['math_complexity'])

print("🏆 TOP 10 MOST COMPLEX MATHEMATICAL FUNCTIONS:")
print("-" * 60)
for func in math_functions[:10]:
    print(f"Level {func['level']:2}, Complexity {func['math_complexity']:3}: {func['name']}")
    print(f"  Args: {', '.join(func['args'][:5])}")
    print(f"  Operations: {func['num_binops']} binops, {func['num_calls']} calls")
    print(f"  {func['file']}:{func['line']}")
    print()

print("🎯 TOP 10 SIMPLEST MATHEMATICAL FUNCTIONS:")
print("-" * 60)
for func in math_functions[-10:]:
    print(f"Level {func['level']:2}, Complexity {func['math_complexity']:3}: {func['name']}")
    print(f"  Args: {', '.join(func['args'][:5])}")
    print(f"  {func['file']}:{func['line']}")

print()

# Functions with 71
math_71 = [f for f in math_functions if f['has_71']]
print(f"📊 MATHEMATICAL FUNCTIONS WITH 71: {len(math_71)}")
print("-" * 60)
for func in math_71[:10]:
    print(f"Level {func['level']:2}, Complexity {func['math_complexity']:3}: {func['name']}")
    print(f"  {func['file']}:{func['line']}")

print()

# Categorize by mathematical domain
print("📊 CATEGORIZATION BY MATHEMATICAL DOMAIN:")
print("-" * 60)

domains = defaultdict(list)
for func in math_functions:
    name_lower = func['name'].lower()
    
    if any(word in name_lower for word in ['hecke', 'eigenvalue', 'operator']):
        domains['hecke_theory'].append(func)
    elif any(word in name_lower for word in ['conductor', 'discriminant', 'level']):
        domains['arithmetic'].append(func)
    elif any(word in name_lower for word in ['dimension', 'rank', 'genus']):
        domains['geometry'].append(func)
    elif any(word in name_lower for word in ['fourier', 'coefficient', 'expansion']):
        domains['analysis'].append(func)
    elif any(word in name_lower for word in ['prime', 'factor', 'divisor', 'gcd']):
        domains['number_theory'].append(func)
    else:
        domains['other'].append(func)

for domain, funcs in sorted(domains.items(), key=lambda x: -len(x[1])):
    print(f"{domain:20}: {len(funcs):4} functions")

print()

# Save results
output = {
    'total_functions': len(math_functions),
    'functions_with_71': len(math_71),
    'domains': {k: len(v) for k, v in domains.items()},
    'functions': math_functions[:500],  # Limit to 500
    'stats': {
        'avg_complexity': sum(f['math_complexity'] for f in math_functions) / len(math_functions),
        'max_complexity': max(f['math_complexity'] for f in math_functions),
        'min_complexity': min(f['math_complexity'] for f in math_functions),
        'avg_args': sum(f['num_args'] for f in math_functions) / len(math_functions),
        'levels_used': len(set(f['level'] for f in math_functions))
    }
}

with open('lmfdb_math_functions.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"💾 Saved to: lmfdb_math_functions.json")
print()

# Generate mathematical models
print("🔮 GENERATING MATHEMATICAL MODELS:")
print("-" * 60)

models = []

for func in math_functions[:20]:  # Model top 20
    # Create mathematical signature
    args_str = ', '.join(func['args']) if func['args'] else 'void'
    
    # Classify operations
    op_types = defaultdict(int)
    for op in func['operations']:
        if op['type'] == 'binop':
            op_types[op['operation']] += 1
        else:
            op_types[op['function']] += 1
    
    model = {
        'name': func['name'],
        'signature': f"{func['name']}({args_str})",
        'complexity': func['math_complexity'],
        'level': func['level'],
        'operations': dict(op_types),
        'mathematical_form': None  # To be filled
    }
    
    # Infer mathematical form
    if 'Mod' in op_types or 'mod' in op_types:
        model['mathematical_form'] = 'modular_arithmetic'
    elif 'Pow' in op_types or 'pow' in op_types:
        model['mathematical_form'] = 'exponential'
    elif 'Mult' in op_types and 'Add' in op_types:
        model['mathematical_form'] = 'polynomial'
    elif 'Div' in op_types or 'FloorDiv' in op_types:
        model['mathematical_form'] = 'rational'
    else:
        model['mathematical_form'] = 'linear'
    
    models.append(model)
    
    print(f"{model['signature']}")
    print(f"  Form: {model['mathematical_form']}")
    print(f"  Level: {model['level']}, Complexity: {model['complexity']}")
    print(f"  Operations: {dict(list(op_types.items())[:3])}")
    print()

# Save models
with open('lmfdb_math_models.json', 'w') as f:
    json.dump(models, f, indent=2)

print(f"💾 Saved models to: lmfdb_math_models.json")
print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()
print(f"Total mathematical functions: {len(math_functions)}")
print(f"Functions with 71: {len(math_71)}")
print(f"Avg complexity: {output['stats']['avg_complexity']:.1f}")
print(f"Levels used: {output['stats']['levels_used']}/71")
print()
print("Domains:")
for domain, count in sorted(domains.items(), key=lambda x: -len(x[1]))[:5]:
    print(f"  {domain}: {len(count)}")
print()
print("✅ MATHEMATICAL EXTRACTION COMPLETE")
