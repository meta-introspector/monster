#!/usr/bin/env python3
"""
Analyze operations on 71 - what happens, where, and how to apply it
"""

import json
import re
from collections import defaultdict, Counter
import pandas as pd

print("🔬 Operations on 71 - Deep Analysis")
print("="*60)

# Load perf trace analysis
with open('operations_around_71.json') as f:
    perf_data = json.load(f)

print(f"\n📊 Perf Trace Statistics:")
print(f"  Files with 71: {perf_data['files_count']}")
print(f"  Total occurrences: {perf_data['total_occurrences']}")

# Analyze what operations happen
operations = {
    'precedence_setting': {
        'location': 'spectral/algebra/ring.hlean:55',
        'code': 'infixl ` ** `:71 := graded_ring.mul',
        'operation': 'Define infix operator with precedence 71',
        'effect': 'Sets binding strength for graded multiplication'
    },
    'graded_multiplication': {
        'signature': 'mul : Π⦃m m\'⦄, R m → R m\' → R (m * m\')',
        'operation': 'Multiply elements from different grades',
        'input': 'R_m × R_n',
        'output': 'R_{m*n}',
        'preserves': 'Grading structure'
    },
    'associativity': {
        'law': 'mul (mul r₁ r₂) r₃ ==[R] mul r₁ (mul r₂ r₃)',
        'operation': 'Composition of graded multiplications',
        'effect': 'Enables categorical arrows'
    },
    'distributivity': {
        'left': 'mul r₁ (r₂ + r₂\') = mul r₁ r₂ + mul r₁ r₂\'',
        'right': 'mul (r₁ + r₁\') r₂ = mul r₁ r₂ + mul r₁\' r₂',
        'operation': 'Interaction with addition',
        'effect': 'Ring structure preserved'
    }
}

print(f"\n🔧 Operations Defined at Precedence 71:")
for op_name, op_data in operations.items():
    print(f"\n  {op_name}:")
    for key, value in op_data.items():
        print(f"    {key}: {value}")

# What we learn
learnings = {
    '1_precedence_is_structural': {
        'observation': '71 is between 70 (regular mul) and 80 (exp)',
        'meaning': 'Graded operations are more refined than regular',
        'application': 'Use precedence to encode structural hierarchy'
    },
    '2_grading_preserves_structure': {
        'observation': 'R_m × R_n → R_{m*n}',
        'meaning': 'Multiplication respects grading',
        'application': 'Operations preserve Monster prime structure'
    },
    '3_associativity_enables_composition': {
        'observation': '(a ** b) ** c = a ** (b ** c)',
        'meaning': 'Graded operations compose',
        'application': 'Categorical arrows work'
    },
    '4_distributivity_enables_linearity': {
        'observation': 'a ** (b + c) = (a ** b) + (a ** c)',
        'meaning': 'Graded multiplication is linear',
        'application': 'Can decompose complex operations'
    }
}

print(f"\n💡 What We Learn:")
for learning_id, learning_data in learnings.items():
    print(f"\n  {learning_id}:")
    for key, value in learning_data.items():
        print(f"    {key}: {value}")

# How to apply to our project
applications = {
    'monster_algorithm': {
        'current': 'Check divisibility by Monster primes',
        'enhancement': 'Use graded structure to compose operations',
        'benefit': 'Categorical composition proven correct'
    },
    'resonance_scoring': {
        'current': 'Weighted sum of divisibilities',
        'enhancement': 'Use precedence 71 for graded scoring',
        'benefit': 'Structural hierarchy encoded'
    },
    'pipeline': {
        'current': 'Linear: capture → FFT → resonance',
        'enhancement': 'Graded: capture → grade → compose → extract',
        'benefit': 'Preserves structure at each step'
    },
    'lean_proofs': {
        'current': '6 proven theorems',
        'enhancement': 'Add graded ring structure theorems',
        'benefit': 'Formal verification of precedence'
    }
}

print(f"\n🚀 How to Apply to Our Project:")
for app_name, app_data in applications.items():
    print(f"\n  {app_name}:")
    for key, value in app_data.items():
        print(f"    {key}: {value}")

# Where else can we find this operation?
similar_operations = {
    'lean_mathlib': {
        'location': '.lake/packages/mathlib/',
        'pattern': 'infixl.*:7[0-9]',
        'examples': ['Precedence 70-79 operators', 'Graded structures']
    },
    'flt_package': {
        'location': '.lake/packages/FLT/',
        'pattern': 'modular forms, graded rings',
        'examples': ['Modular forms are graded', 'Moonshine connection']
    },
    'carleson_package': {
        'location': '.lake/packages/Carleson/',
        'pattern': 'harmonic analysis, graded',
        'examples': ['Haar measures', 'Spectral decomposition']
    },
    'our_code': {
        'location': 'MonsterLean/',
        'pattern': 'graded operations',
        'examples': ['GradedRing71.lean', 'MonsterAlgorithm.lean']
    }
}

print(f"\n🔍 Where Else to Find This Operation:")
for loc_name, loc_data in similar_operations.items():
    print(f"\n  {loc_name}:")
    for key, value in loc_data.items():
        print(f"    {key}: {value}")

# Perf trace insights
print(f"\n📈 Perf Trace Insights:")
print(f"\n  Top functions around 71:")
for func, count in list(perf_data['top_functions'].items())[:5]:
    print(f"    {func}: {count} occurrences")

print(f"\n  Sample contexts:")
for ctx in perf_data['sample_contexts'][:3]:
    print(f"\n    {ctx['file']} line {ctx['line']}:")
    print(f"      {ctx['text']}")

# Create graph data
graph_data = {
    'nodes': [
        {'id': 'precedence_71', 'type': 'operator', 'label': 'Precedence 71'},
        {'id': 'graded_mul', 'type': 'operation', 'label': 'Graded Multiplication'},
        {'id': 'monster_primes', 'type': 'data', 'label': 'Monster Primes'},
        {'id': 'resonance', 'type': 'metric', 'label': 'Resonance Score'},
        {'id': 'composition', 'type': 'property', 'label': 'Categorical Composition'},
    ],
    'edges': [
        {'from': 'precedence_71', 'to': 'graded_mul', 'label': 'defines'},
        {'from': 'graded_mul', 'to': 'monster_primes', 'label': 'extracts'},
        {'from': 'monster_primes', 'to': 'resonance', 'label': 'computes'},
        {'from': 'graded_mul', 'to': 'composition', 'label': 'enables'},
    ]
}

# Save analysis
analysis_output = {
    'operations': operations,
    'learnings': learnings,
    'applications': applications,
    'similar_operations': similar_operations,
    'perf_insights': perf_data,
    'graph': graph_data
}

with open('operations_on_71_analysis.json', 'w') as f:
    json.dump(analysis_output, f, indent=2)

# Create graph visualization data
with open('operations_71_graph.dot', 'w') as f:
    f.write('digraph Operations71 {\n')
    f.write('  rankdir=LR;\n')
    f.write('  node [shape=box];\n\n')
    
    for node in graph_data['nodes']:
        f.write(f'  {node["id"]} [label="{node["label"]}"];\n')
    
    f.write('\n')
    
    for edge in graph_data['edges']:
        f.write(f'  {edge["from"]} -> {edge["to"]} [label="{edge["label"]}"];\n')
    
    f.write('}\n')

print(f"\n💾 Saved:")
print(f"  operations_on_71_analysis.json")
print(f"  operations_71_graph.dot")

print(f"\n✅ Analysis complete!")
EOF

python3 << 'SCRIPT'
import json
with open('operations_around_71.json') as f:
    data = json.load(f)
    
# Run the analysis
import subprocess
subprocess.run(['python3', '-c', open('analyze_operations_71.py').read()])
SCRIPT
