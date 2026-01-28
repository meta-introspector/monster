# LMFDB Prime 71 Objects: Complexity Analysis & Topological Sort

## Overview

We have analyzed **41 objects** containing prime 71 from the LMFDB codebase, assigned complexity levels (1-71), and topologically sorted them by dependencies.

## Complexity Scoring System

### Formula
```
total_complexity = base_complexity + file_complexity + line_complexity + code_complexity

where:
- base_complexity: Type-specific base (1-71)
- file_complexity: len(file_path) / 10
- line_complexity: line_number / 100
- code_complexity: len(code) / 50

level = (total_complexity % 71) + 1
```

### Object Type Hierarchy (Base Complexity)

#### Level 1-10: Basic Objects
- **Prime** (1): Prime number 71
- **Constant** (2): Literal constant 71

#### Level 11-20: Arithmetic Objects
- **Conductor** (11): Conductor = 71
- **Discriminant** (12): Discriminant = 71
- **Level** (13): Level = 71
- **Degree** (14): Degree = 71

#### Level 21-30: Geometric Objects
- **Dimension** (21): Dimension = 71
- **Genus** (22): Genus = 71
- **Rank** (23): Rank = 71

#### Level 31-40: Field Objects
- **Field Size** (31): Finite field F_71
- **Field Extension** (32): Field extension degree 71

#### Level 41-50: Analytic Objects
- **Coefficient** (41): Fourier coefficient a_71
- **Eigenvalue** (42): Hecke eigenvalue λ_71
- **Hecke** (43): Hecke operator T_71

#### Level 51-60: Modular Objects
- **Modular Form** (51): Modular form with 71
- **Elliptic Curve** (52): Elliptic curve with 71
- **Abelian Variety** (53): Abelian variety over F_71

#### Level 61-71: Complex Objects
- **Hypergeometric** (61): Hypergeometric motive
- **Number Field** (62): Number field with 71
- **Galois Group** (63): Galois group of order 71

## Statistics

### Overall
- **Total objects**: 41
- **Levels used**: 23/71 (32% coverage)
- **Min complexity**: 5
- **Max complexity**: 56
- **Avg complexity**: 22.1

### Distribution
- **Most populated level**: 24 (5 objects)
- **Levels with 4+ objects**: 7, 8, 23, 24
- **Unique levels**: 23

## Top 5 Most Complex Objects

| Level | Complexity | Type        | Location                                      |
|-------|------------|-------------|-----------------------------------------------|
| 57    | 56         | eigenvalue  | modular_curves/main.py:1031                   |
| 52    | 51         | eigenvalue  | hilbert_modular_forms/hilbert_modular_form.py:147 |
| 52    | 51         | eigenvalue  | hilbert_modular_forms/hilbert_modular_form.py:146 |
| 51    | 50         | eigenvalue  | modl_galois_representations/main.py:152       |
| 48    | 47         | coefficient | abvar/fq/test_browse_page.py:298              |

**Insight**: Most complex objects are **eigenvalues** in modular forms and Galois representations.

## Top 5 Simplest Objects

| Level | Complexity | Type  | Location                              |
|-------|------------|-------|---------------------------------------|
| 6     | 5          | prime | abvar/fq/main.py:269                  |
| 6     | 5          | prime | number_fields/test_numberfield.py:88  |
| 7     | 6          | prime | abvar/fq/test_browse_page.py:176      |
| 7     | 6          | prime | abvar/fq/test_browse_page.py:177      |
| 7     | 6          | prime | belyi/test_belyi.py:188               |

**Insight**: Simplest objects are **prime literals** in test files and main modules.

## Level Distribution

### Top 10 Levels by Object Count

| Level | Objects | Dominant Type |
|-------|---------|---------------|
| 24    | 5       | Mixed         |
| 7     | 4       | prime         |
| 8     | 4       | prime/collection |
| 23    | 4       | Mixed         |
| 6     | 2       | prime         |
| 10    | 2       | collection/prime |
| 13    | 2       | collection/prime |
| 48    | 2       | coefficient   |
| 52    | 2       | eigenvalue    |
| 9     | 1       | dict_value    |

## Topological Sort

Objects are sorted by dependencies:
1. **Simpler objects** (primes, constants) come first
2. **Arithmetic objects** (conductors, discriminants) depend on primes
3. **Geometric objects** (dimensions, genus) depend on arithmetic
4. **Analytic objects** (eigenvalues, coefficients) depend on geometric
5. **Complex objects** (modular forms, varieties) depend on all above

### Dependency Graph

```
Level 1-10 (Basic)
    ↓
Level 11-20 (Arithmetic)
    ↓
Level 21-30 (Geometric)
    ↓
Level 31-40 (Fields)
    ↓
Level 41-50 (Analytic)
    ↓
Level 51-60 (Modular)
    ↓
Level 61-71 (Complex)
```

## Objects by Mathematical Category

### Abelian Varieties (Level 53)
- **Location**: `abvar/fq/`
- **Complexity**: 5-47
- **Count**: 8 objects
- **Key object**: Abelian variety over F_71 (Dim 2, slopes [0, 1/2, 1/2, 1])

### Eigenvalues (Level 42)
- **Location**: `hilbert_modular_forms/`, `modular_curves/`
- **Complexity**: 50-56
- **Count**: 4 objects
- **Key object**: Hecke eigenvalue λ_71

### Coefficients (Level 41)
- **Location**: `abvar/fq/test_browse_page.py`
- **Complexity**: 47
- **Count**: 2 objects
- **Key object**: Fourier coefficient a_71

### Conductors (Level 11)
- **Location**: `elliptic_curves/test_browse_page.py`
- **Complexity**: 17
- **Count**: 1 object
- **Key object**: Elliptic curve conductor = 71

### Prime Lists (Level 1)
- **Location**: `hypergm/main.py`, test files
- **Complexity**: 5-13
- **Count**: 16 objects
- **Key object**: Prime list [2, 3, 5, ..., 71]

## Complexity Insights

### By File Type
- **Test files** (`test_*.py`): Complexity 5-47 (simple to moderate)
- **Main modules** (`main.py`): Complexity 5-56 (full range)
- **Core modules** (e.g., `hilbert_modular_form.py`): Complexity 50-51 (high)

### By Object Type
- **Primes**: Complexity 5-13 (simplest)
- **Collections**: Complexity 7-12 (simple)
- **Conductors**: Complexity 17 (moderate)
- **Coefficients**: Complexity 47 (high)
- **Eigenvalues**: Complexity 50-56 (highest)

### Complexity Factors
1. **File depth**: Deeper files → higher complexity
2. **Line number**: Later lines → higher complexity
3. **Code length**: Longer code → higher complexity
4. **Object type**: Analytic objects → higher base complexity

## Usage

### Load Complexity Data
```python
import json

with open('lmfdb_71_complexity.json') as f:
    data = json.load(f)

# Get objects by level
level_24 = [obj for obj in data['objects'] if obj['level'] == 24]

# Get topological order
topo_order = data['topological_order']

# Get most complex objects
sorted_objs = sorted(data['objects'], key=lambda x: -x['total_complexity'])
```

### Query by Complexity Range
```python
# Objects with complexity 40-50 (analytic objects)
analytic = [obj for obj in data['objects'] 
            if 40 <= obj['total_complexity'] <= 50]

# Objects with complexity < 10 (basic objects)
basic = [obj for obj in data['objects'] 
         if obj['total_complexity'] < 10]
```

### Filter by Type
```python
# All eigenvalue objects
eigenvalues = [obj for obj in data['objects'] 
               if obj['type'] == 'eigenvalue']

# All prime objects
primes = [obj for obj in data['objects'] 
          if obj['type'] == 'prime']
```

## Key Findings

### 1. Eigenvalues are Most Complex
- Highest complexity: 50-56
- Located in core modular form modules
- Require deep mathematical context

### 2. Primes are Simplest
- Lowest complexity: 5-13
- Located in test files and utilities
- Self-contained, no dependencies

### 3. Level Distribution is Sparse
- Only 23/71 levels used (32%)
- Most objects cluster in levels 6-24
- High-complexity objects (50+) are rare

### 4. Topological Structure is Clear
- Basic → Arithmetic → Geometric → Analytic → Complex
- Dependencies follow mathematical hierarchy
- No circular dependencies detected

### 5. Abelian Varieties Span Range
- Complexity 5-47 (wide range)
- Both simple (prime lists) and complex (coefficients)
- Central to LMFDB structure

## Next Steps

1. ✅ Apply Hecke operators to each level
2. ✅ Compute eigenvalues for each object
3. ✅ Create level-based Parquet shards
4. ✅ Visualize dependency graph
5. ✅ Export to HuggingFace with complexity metadata

## References

- Complexity theory: Kolmogorov complexity
- Topological sort: Kahn's algorithm
- Dependency analysis: Graph theory
- LMFDB structure: Mathematical database design
