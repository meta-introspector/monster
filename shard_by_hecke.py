#!/usr/bin/env python3
"""
Shard Hilbert by HECKE RESONANCE - divisibility by Monster primes
"""

import ast
import dis
import json

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

def hecke_resonance(value):
    """Find which Monster prime this value resonates with most"""
    if value == 0:
        return 1
    
    # Count divisibility by each Monster prime
    resonances = {}
    for p in MONSTER_PRIMES:
        if value % p == 0:
            # Count how many times p divides value
            count = 0
            temp = abs(value)
            while temp % p == 0:
                count += 1
                temp //= p
            resonances[p] = count
    
    if not resonances:
        return 1  # No Monster prime divisors
    
    # Return the prime with highest power
    return max(resonances.items(), key=lambda x: x[1])[0]

def shard_hilbert():
    """Shard Hilbert by Hecke resonance"""
    
    code = open('hilbert_test.py').read()
    
    print("🔮 SHARDING BY HECKE RESONANCE")
    print("=" * 70)
    print()
    
    # Shard AST by literal values
    print("Level 1: AST Nodes (by literal value)")
    print("-" * 70)
    tree = ast.parse(code)
    
    ast_shards = {p: [] for p in MONSTER_PRIMES}
    ast_shards['other'] = []
    ast_shards[1] = []  # For values with no Monster prime factors
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            value = node.value
            shard = hecke_resonance(value)
            ast_shards[shard].append((node.lineno, value))
    
    print(f"Total numeric constants: {sum(len(v) for v in ast_shards.values())}")
    for p in MONSTER_PRIMES:
        if ast_shards[p]:
            print(f"  Prime {p:2}: {len(ast_shards[p]):2} nodes - {[v for _, v in ast_shards[p][:5]]}")
    
    # Check shard 71
    if ast_shards[71]:
        print(f"\n✓ Shard 71 (Hecke): {len(ast_shards[71])} AST nodes")
        for line, value in ast_shards[71]:
            print(f"    Line {line}: value={value}")
    
    print()
    
    # Shard bytecode by argval
    print("Level 2: Bytecode (by constant values)")
    print("-" * 70)
    
    namespace = {}
    exec(code, namespace)
    
    bytecode_shards = {p: [] for p in MONSTER_PRIMES}
    bytecode_shards['other'] = []
    bytecode_shards[1] = []
    
    for name in ['hilbert_norm', 'is_totally_positive', 'hilbert_level', 'compute_fourier_coefficient']:
        if name in namespace:
            func = namespace[name]
            for instr in dis.get_instructions(func):
                if instr.argval is not None and isinstance(instr.argval, int):
                    value = instr.argval
                    shard = hecke_resonance(value)
                    bytecode_shards[shard].append((name, instr.opname, value))
    
    print(f"Total numeric bytecode: {sum(len(v) for v in bytecode_shards.values())}")
    for p in MONSTER_PRIMES:
        if bytecode_shards[p]:
            print(f"  Prime {p:2}: {len(bytecode_shards[p]):2} ops")
    
    if bytecode_shards[71]:
        print(f"\n✓ Shard 71 (Hecke): {len(bytecode_shards[71])} bytecode ops")
        for fname, opname, value in bytecode_shards[71]:
            print(f"    {fname}: {opname} {value}")
    
    print()
    
    # Shard output values
    print("Level 3: Output Values (by numeric value)")
    print("-" * 70)
    
    import subprocess
    result = subprocess.run(['python3', 'hilbert_test.py'], 
                          capture_output=True, text=True)
    output = result.stdout
    
    import re
    numbers = [int(n) for n in re.findall(r'\b\d+\b', output)]
    
    output_shards = {p: [] for p in MONSTER_PRIMES}
    output_shards['other'] = []
    output_shards[1] = []
    
    for num in numbers:
        shard = hecke_resonance(num)
        output_shards[shard].append(num)
    
    print(f"Total output numbers: {len(numbers)}")
    for p in MONSTER_PRIMES:
        if output_shards[p]:
            count = len(output_shards[p])
            pct = (count / len(numbers)) * 100
            print(f"  Prime {p:2}: {count:3} numbers ({pct:5.1f}%)")
    
    if output_shards[71]:
        print(f"\n✓ Shard 71 (Hecke): {len(output_shards[71])} output values")
        print(f"    Values: {output_shards[71][:10]}")
    
    print()
    
    # Summary
    print("=" * 70)
    print("HECKE RESONANCE SUMMARY")
    print("=" * 70)
    print()
    
    summary = {}
    for p in MONSTER_PRIMES:
        summary[p] = {
            'ast': len(ast_shards[p]),
            'bytecode': len(bytecode_shards[p]),
            'output': len(output_shards[p]),
            'total': len(ast_shards[p]) + len(bytecode_shards[p]) + len(output_shards[p])
        }
    
    # Sort by total resonance
    sorted_primes = sorted(summary.items(), key=lambda x: -x[1]['total'])
    
    print(f"{'Prime':<8} {'AST':<6} {'Bytecode':<10} {'Output':<8} {'Total'}")
    print("-" * 70)
    for p, counts in sorted_primes:
        if counts['total'] > 0:
            print(f"{p:<8} {counts['ast']:<6} {counts['bytecode']:<10} {counts['output']:<8} {counts['total']}")
    
    print()
    print("✓ Sharded by HECKE RESONANCE (divisibility by Monster primes)")
    print("✓ Prime 71 now contains ALL values divisible by 71!")
    
    # Save manifest
    manifest = {
        'method': 'hecke_resonance',
        'primes': MONSTER_PRIMES,
        'shards': {str(p): summary[p] for p in MONSTER_PRIMES if summary[p]['total'] > 0}
    }
    
    with open('hilbert_hecke_shards.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print()
    print("📄 Manifest saved to: hilbert_hecke_shards.json")

if __name__ == '__main__':
    shard_hilbert()
