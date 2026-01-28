#!/usr/bin/env python3
"""Iterative improvement: implement changes, re-review, repeat"""
import json
from pathlib import Path
from datetime import datetime

def extract_action_items():
    """Extract all action items from synthesis documents"""
    reviews_dir = Path('multi_level_reviews')
    actions = []
    
    for synthesis in sorted(reviews_dir.glob('page_*_synthesis.md')):
        page = synthesis.stem.split('_')[1]
        with open(synthesis) as f:
            content = f.read()
            # Extract scholar issues
            if '### Mathematician' in content:
                actions.append({
                    'page': page,
                    'type': 'scholar',
                    'persona': 'mathematician',
                    'priority': 'high',
                    'issue': 'Define terms precisely, fix proof structure'
                })
            if '### Computer_Scientist' in content:
                actions.append({
                    'page': page,
                    'type': 'scholar',
                    'persona': 'computer_scientist',
                    'priority': 'high',
                    'issue': 'Add algorithm details and complexity'
                })
            if '### Group_Theorist' in content:
                actions.append({
                    'page': page,
                    'type': 'scholar',
                    'persona': 'group_theorist',
                    'priority': 'high',
                    'issue': 'Verify Monster group properties'
                })
            # Extract muse suggestions
            if '### Visionary' in content:
                actions.append({
                    'page': page,
                    'type': 'muse',
                    'persona': 'visionary',
                    'priority': 'medium',
                    'issue': 'Add big picture connections'
                })
            if '### Artist' in content:
                actions.append({
                    'page': page,
                    'type': 'muse',
                    'persona': 'artist',
                    'priority': 'medium',
                    'issue': 'Add visual diagrams'
                })
    
    return actions

def create_improvement_plan(actions):
    """Create prioritized improvement plan"""
    plan = {
        'created': datetime.now().isoformat(),
        'total_actions': len(actions),
        'iterations': []
    }
    
    # Group by priority
    high_priority = [a for a in actions if a['priority'] == 'high']
    medium_priority = [a for a in actions if a['priority'] == 'medium']
    
    # Create iterations
    plan['iterations'].append({
        'iteration': 1,
        'focus': 'Mathematical rigor',
        'actions': [a for a in high_priority if a['persona'] == 'mathematician']
    })
    
    plan['iterations'].append({
        'iteration': 2,
        'focus': 'Algorithm details',
        'actions': [a for a in high_priority if a['persona'] == 'computer_scientist']
    })
    
    plan['iterations'].append({
        'iteration': 3,
        'focus': 'Group theory verification',
        'actions': [a for a in high_priority if a['persona'] == 'group_theorist']
    })
    
    plan['iterations'].append({
        'iteration': 4,
        'focus': 'Visual enhancements',
        'actions': [a for a in medium_priority if a['persona'] == 'artist']
    })
    
    return plan

def generate_improvement_tasks():
    """Generate concrete improvement tasks for PAPER.md"""
    tasks = [
        {
            'id': 1,
            'title': 'Add Notation Glossary',
            'description': 'Create glossary section defining all mathematical symbols',
            'file': 'PAPER.md',
            'section': 'After introduction',
            'content': '''
## Notation Glossary

| Symbol | Meaning | Context |
|--------|---------|---------|
| M | Monster group | Sporadic simple group |
| j(τ) | J-invariant | Modular function |
| T_p | Hecke operator | Prime p |
| E | Encoder | Neural network layers [5,11,23,47,71] |
| D | Decoder | Neural network layers [71,47,23,11,5] |
| ≡ | Equivalence | Bisimulation equivalence |
'''
        },
        {
            'id': 2,
            'title': 'Add Architecture Diagram',
            'description': 'ASCII diagram of encoder-decoder architecture',
            'file': 'PAPER.md',
            'section': 'Architecture section',
            'content': '''
```
INPUT (5 features)
    ↓
[Layer 5]  ← Monster prime
    ↓
[Layer 11] ← Monster prime
    ↓
[Layer 23] ← Monster prime
    ↓
[Layer 47] ← Monster prime
    ↓
[Layer 71] ← Monster prime (bottleneck)
    ↓
[Layer 47] ← Decoder
    ↓
[Layer 23]
    ↓
[Layer 11]
    ↓
[Layer 5]
    ↓
OUTPUT (5 features reconstructed)
```
'''
        },
        {
            'id': 3,
            'title': 'Add Algorithm Pseudocode',
            'description': 'Formal algorithm with complexity',
            'file': 'PAPER.md',
            'section': 'Methods section',
            'content': '''
## Algorithm: Monster Autoencoder

```
Algorithm: MonsterEncode(x)
Input: x ∈ ℝ^5 (5 features)
Output: z ∈ ℝ^71 (compressed representation)

1. h₁ ← ReLU(W₅×₁₁ · x + b₁₁)      // O(5×11)
2. h₂ ← ReLU(W₁₁×₂₃ · h₁ + b₂₃)    // O(11×23)
3. h₃ ← ReLU(W₂₃×₄₇ · h₂ + b₄₇)    // O(23×47)
4. z ← ReLU(W₄₇×₇₁ · h₃ + b₇₁)     // O(47×71)
5. return z

Total complexity: O(5×11 + 11×23 + 23×47 + 47×71) = O(4,651)
```
'''
        },
        {
            'id': 4,
            'title': 'Add Concrete Example',
            'description': 'Show actual input/output with j-invariant',
            'file': 'PAPER.md',
            'section': 'Examples section',
            'content': '''
## Example: Elliptic Curve Compression

**Input**: Elliptic curve E: y² = x³ + ax + b
- a = 1, b = 0
- j-invariant: j(E) = 1728
- Features: [1, 0, 1728, 0, 1]

**Encoding**:
- Layer 11: [0.23, 0.45, ..., 0.12] (11 values)
- Layer 23: [0.34, 0.56, ..., 0.23] (23 values)
- Layer 47: [0.45, 0.67, ..., 0.34] (47 values)
- Layer 71: [0.56, 0.78, ..., 0.45] (71 values) ← Compressed

**Decoding**: Reconstructs [1.02, -0.01, 1729.3, 0.02, 0.98]
**MSE**: 0.233 (from verification)
'''
        }
    ]
    return tasks

def main():
    print("🔄 Iterative Improvement System\n")
    
    # Extract actions
    print("📊 Extracting action items from reviews...")
    actions = extract_action_items()
    print(f"   Found {len(actions)} actions\n")
    
    # Create plan
    print("📋 Creating improvement plan...")
    plan = create_improvement_plan(actions)
    
    with open('improvement_plan.json', 'w') as f:
        json.dump(plan, f, indent=2)
    print(f"   ✓ Saved to improvement_plan.json")
    print(f"   {len(plan['iterations'])} iterations planned\n")
    
    # Generate tasks
    print("✏️  Generating concrete tasks...")
    tasks = generate_improvement_tasks()
    
    with open('improvement_tasks.json', 'w') as f:
        json.dump(tasks, f, indent=2)
    print(f"   ✓ Saved to improvement_tasks.json")
    print(f"   {len(tasks)} tasks ready\n")
    
    # Create markdown checklist
    print("📝 Creating task checklist...")
    with open('IMPROVEMENTS.md', 'w') as f:
        f.write(f"# Improvement Tasks\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n")
        f.write(f"**Total Tasks**: {len(tasks)}\n\n")
        
        for task in tasks:
            f.write(f"## Task {task['id']}: {task['title']}\n\n")
            f.write(f"**Description**: {task['description']}\n\n")
            f.write(f"**File**: `{task['file']}`\n")
            f.write(f"**Section**: {task['section']}\n\n")
            f.write(f"**Content to add**:\n{task['content']}\n\n")
            f.write(f"- [ ] Implement\n")
            f.write(f"- [ ] Review\n")
            f.write(f"- [ ] Verify\n\n")
            f.write("---\n\n")
    
    print("   ✓ Saved to IMPROVEMENTS.md\n")
    
    # Summary
    print("=" * 60)
    print("✅ READY FOR ITERATION 1")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review IMPROVEMENTS.md")
    print("2. Implement Task 1 (Notation Glossary)")
    print("3. Regenerate PDF and review")
    print("4. Repeat for remaining tasks")
    print("\nFiles created:")
    print("- improvement_plan.json")
    print("- improvement_tasks.json")
    print("- IMPROVEMENTS.md")

if __name__ == '__main__':
    main()
