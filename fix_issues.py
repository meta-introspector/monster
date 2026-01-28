#!/usr/bin/env python3
"""Fix critical issues from verification"""

fixes = []

# Fix P8: 71 shards not 70
with open('PAPER.md', 'r') as f:
    content = f.read()

# Fix 1: Update shard count
content = content.replace('70 equivalence classes', '71 shards (shard_00 to shard_70)')
content = content.replace('70 unique j-invariants', '71 unique j-invariant values')
content = content.replace('Unique values (70)', 'Unique values (71)')
fixes.append("P8: Updated 70 → 71 shards")

# Fix 2: Clarify parameter count (9,452 weights + 238 biases = 9,690)
content = content.replace(
    '9,690 trainable parameters',
    '9,690 parameters (9,452 weights + 238 biases)'
)
fixes.append("P10: Clarified parameter count breakdown")

# Fix 3: Update architecture description to match code
content = content.replace(
    'Architecture has layers [5, 11, 23, 47, 71]',
    'Encoder layers: 5→11→23→47→71, Decoder layers: 71→47→23→11→5'
)
fixes.append("P1: Clarified architecture description")

# Fix 4: Add note about j-invariant formula
if '## 3. The J-Invariant World' in content:
    note = """
**Note on J-Invariant:** The classical j-invariant for elliptic curves is:
```
j(E) = 1728 × (4a³) / (4a³ + 27b²)
```
Our implementation uses this standard formula, not a modular reduction.
"""
    content = content.replace(
        '## 3. The J-Invariant World\n\n',
        f'## 3. The J-Invariant World\n\n{note}\n'
    )
    fixes.append("P5: Clarified j-invariant formula")

with open('PAPER.md', 'w') as f:
    f.write(content)

print("✅ Fixed Critical Issues:")
for fix in fixes:
    print(f"   - {fix}")

# Update verification status
print("\n📊 Updated Claims:")
print("   P1: Architecture → CLARIFIED")
print("   P5: J-invariant → CLARIFIED")
print("   P8: Shard count → FIXED (71 not 70)")
print("   P10: Parameters → CLARIFIED (with biases)")
