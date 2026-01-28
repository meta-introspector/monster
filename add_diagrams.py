#!/usr/bin/env python3
"""Add ASCII diagrams to PAPER.md"""

ARCH_DIAGRAM = """
```
┌─────────────────────────────────────────┐
│         MONSTER AUTOENCODER             │
├─────────────────────────────────────────┤
│  INPUT: [a,b,c,d,e] ∈ R^5              │
│     ↓                                   │
│  [W_11]  Monster Prime: 11              │
│     ↓    σ(W·x + b) → R^11             │
│  [W_23]  Monster Prime: 23              │
│     ↓    σ(W·h + b) → R^23             │
│  [W_47]  Monster Prime: 47              │
│     ↓    σ(W·h + b) → R^47             │
│  [W_71]  Monster Prime: 71 (MAX)        │
│     ↓    BOTTLENECK → R^71             │
│  [DECODER: 71→47→23→11→5]              │
│     ↓                                   │
│  OUTPUT: [a',b',c',d',e'] ∈ R^5        │
│  MSE = 0.233                            │
└─────────────────────────────────────────┘
```
"""

J_DIAGRAM = """
```
LMFDB (7,115 objects)
        ↓
Extract j-invariants
        ↓
Unique values (70)
        ↓
┌──────────────────┐
│ Shard by j-value │
│ shard_00 ... _70 │
└──────────────────┘
        ↓
Encode to R^71
        ↓
23× compression
253,581× overcapacity
```
"""

HECKE_DIAGRAM = """
```
T_2  T_3  T_5  T_7 ... T_71
 ↓    ↓    ↓    ↓       ↓
┌────────────────────────┐
│  Monster Group Space   │
│  Preserves symmetry    │
│  T_p ∘ T_q = T_pq      │
└────────────────────────┘
         ↓
   Neural Network
   respects this!
```
"""

with open('PAPER.md', 'r') as f:
    content = f.read()

# Add architecture diagram
content = content.replace(
    '### 2.1 The 71-Layer Autoencoder\n\n```',
    f'### 2.1 The 71-Layer Autoencoder\n\n{ARCH_DIAGRAM}\n\n**Detailed Structure:**\n\n```'
)

# Add j-invariant diagram
if '## 3. The J-Invariant World' in content:
    content = content.replace(
        '## 3. The J-Invariant World',
        f'## 3. The J-Invariant World\n\n{J_DIAGRAM}'
    )

# Add Hecke diagram
if '### 4.1 Hecke Operators' in content:
    content = content.replace(
        '### 4.1 Hecke Operators',
        f'### 4.1 Hecke Operators\n\n{HECKE_DIAGRAM}'
    )

with open('PAPER.md', 'w') as f:
    f.write(content)

print("✅ Added 3 ASCII diagrams to PAPER.md")
print("   - Architecture diagram")
print("   - J-invariant compression")
print("   - Hecke operators")
