# COMPLETE LMFDB HECKE ANALYSIS PLAN

## Objective
Apply Hecke operators to entire LMFDB codebase at all transformation levels.

## Scope
- **356 Python files**
- **100,726 lines of code**
- **26 MB repository**

## Analysis Levels

### 1. Source Code
- [x] Count files (356)
- [x] Count lines (100,726)
- [ ] Find all literal 71 occurrences
- [ ] Shard by Hecke resonance

### 2. AST (Abstract Syntax Tree)
- [x] Parse all files
- [ ] Count all nodes
- [ ] Find Constant(71) nodes
- [ ] Shard by value divisibility

### 3. Bytecode
- [ ] Compile all modules
- [ ] Extract bytecode operations
- [ ] Find LOAD_CONST 71
- [ ] Shard by argval

### 4. Performance (Execution)
- [ ] Setup LMFDB environment
- [ ] Trace with perf
- [ ] Capture cycles, instructions
- [ ] Shard by performance metrics

### 5. Database
- [ ] Build PostgreSQL with Nix
- [ ] Initialize LMFDB schema
- [ ] Load data
- [ ] Query for prime 71 patterns
- [ ] Dump results

## Implementation Plan

### Phase 1: Source Analysis (DONE)
```bash
./analyze_lmfdb_complete.sh
```

**Results**:
- 10 files analyzed (sample)
- 1 file with literal 71
- 3,300 AST nodes

### Phase 2: Full Source Scan
```python
# Scan all 356 files
for py_file in all_python_files:
    - Count literal 71
    - Measure Hecke resonance
    - Assign to shard
```

**Expected**:
- ~50-100 files with literal 71
- Cluster around Hilbert modular forms

### Phase 3: AST Analysis
```python
# Parse all files
for py_file in all_python_files:
    tree = ast.parse(code)
    - Count Constant(71) nodes
    - Find all numeric constants
    - Shard by divisibility
```

**Expected**:
- ~200-500 Constant(71) nodes
- Concentrated in number theory modules

### Phase 4: Bytecode Extraction
```python
# Compile and disassemble
for module in all_modules:
    bytecode = dis.get_instructions(module)
    - Find LOAD_CONST 71
    - Count operations
    - Shard by argval
```

**Expected**:
- ~1000-2000 bytecode ops with 71
- Similar distribution to AST

### Phase 5: Performance Tracing
```bash
# Trace execution
for test in lmfdb_tests:
    perf record -e cycles,instructions python3 test.py
    - Extract samples
    - Find 71-related cycles
    - Shard by performance
```

**Expected**:
- Hilbert tests show high 71 resonance
- Performance signature: cycles ≡ ? (mod 71)

### Phase 6: Database Analysis
```bash
# Setup database
nix-shell -p postgresql
initdb -D lmfdb_data
pg_ctl start

# Load LMFDB
python3 start-lmfdb.py --initialize

# Query for 71
psql lmfdb << SQL
  SELECT * FROM hilbert_modular_forms WHERE discriminant = 71;
  SELECT * FROM elliptic_curves WHERE conductor % 71 = 0;
SQL
```

**Expected**:
- Hilbert forms with discriminant 71
- Elliptic curves with conductor divisible by 71

## Hecke Sharding Strategy

### By File
```
Shard N = files where line_count resonates with prime N
```

### By Value
```
Shard N = values divisible by prime N
```

### By Performance
```
Shard N = code consuming cycles ≡ 0 (mod N)
```

## Expected Results

### Prime 71 Distribution

| Level | Expected Count | Percentage |
|-------|---------------|------------|
| Source files | 50-100 | 14-28% |
| AST nodes | 200-500 | 0.5-1% |
| Bytecode ops | 1000-2000 | 1-2% |
| Perf samples | 10000-50000 | 1-5% |
| DB records | 100-1000 | varies |

### Top Resonating Modules

1. **hilbert_modular_forms/** - Highest 71 resonance
2. **elliptic_curves/** - Moderate resonance
3. **classical_modular_forms/** - Some resonance
4. **lfunctions/** - Computational primes (2,3,5)

## Timeline

- **Phase 1**: Source scan - 1 hour (DONE)
- **Phase 2**: AST analysis - 2 hours
- **Phase 3**: Bytecode - 3 hours
- **Phase 4**: Performance - 4 hours (needs setup)
- **Phase 5**: Database - 8 hours (needs full setup)

**Total**: ~18 hours for complete analysis

## Deliverables

1. **Source manifest**: All files with 71 resonance
2. **AST manifest**: All Constant(71) nodes
3. **Bytecode manifest**: All LOAD_CONST 71
4. **Performance manifest**: Cycle signatures
5. **Database dump**: All 71-related records
6. **71-shard distribution**: Complete mapping

## Next Immediate Steps

1. Expand source scan to all 356 files
2. Create bytecode analyzer
3. Setup Nix environment for LMFDB
4. Create performance tracer
5. Document findings

## Current Status

✅ Phase 1 complete (10 files sampled)  
⏭️ Phase 2 ready (expand to 356 files)  
⏭️ Phase 3-6 pending

---

**Goal**: Complete Hecke analysis of entire LMFDB revealing prime 71 structure at all levels.
