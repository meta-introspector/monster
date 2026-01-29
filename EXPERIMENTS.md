# 🧪 Monster Group Experiments

**Date**: 2026-01-29  
**Environment**: Nix with GAP + packages  
**Status**: ✅ Ready to run

## Overview

Automated experiments to verify Monster group properties using GAP and capture results in Parquet format.

## Setup

### Enter Nix Environment

```bash
nix-shell shell-experiments.nix
```

This provides:
- GAP with atlasrep and ctbllib packages
- Python with pandas and pyarrow
- All experiment scripts

## Experiments

### 01: Monster Group Order

**File**: `experiments/01_monster_order.sh`

**What it does**:
- Loads Monster group from Atlas
- Computes Order(M)
- Verifies against expected value
- Exports to JSON and Parquet

**Expected**: 808017424794512875886459904961710757005754368000000000

### 02: Conjugacy Classes

**File**: `experiments/02_conjugacy_classes.sh`

**What it does**:
- Loads Monster character table
- Counts conjugacy classes
- Verifies 194 classes
- Exports results

**Expected**: 194 conjugacy classes

### 03: BacktrackKit Test

**File**: `experiments/03_backtrack_test.sh`

**What it does**:
- Tests BacktrackKit conjugacy algorithm
- Uses S5 (symmetric group on 5 elements)
- Verifies conjugacy detection
- Exports results

**Expected**: Algorithm correctly identifies conjugate elements

## Running Experiments

### Run All Experiments

```bash
nix-shell shell-experiments.nix --run ./run_experiments.sh
```

### Run Individual Experiment

```bash
nix-shell shell-experiments.nix

./experiments/01_monster_order.sh
./experiments/02_conjugacy_classes.sh
./experiments/03_backtrack_test.sh
```

## Output Format

### JSON Output

Each experiment produces JSON:
```json
{
  "experiment": "01_monster_order",
  "result": "PASS",
  "order": "808017424794512875886459904961710757005754368000000000",
  "expected": "808017424794512875886459904961710757005754368000000000",
  "match": true
}
```

### Parquet Output

Converted to Parquet for analysis:
```
experiment | result | order | expected | match | timestamp
-----------|--------|-------|----------|-------|----------
01_...     | PASS   | 808.. | 808...   | true  | 2026-...
```

### Combined Results

All experiments combined in:
```
experiments/results/all_experiments.parquet
```

## Results Location

```
experiments/
├── 01_monster_order.sh
├── 02_conjugacy_classes.sh
├── 03_backtrack_test.sh
└── results/
    ├── 01_monster_order.json
    ├── 01_monster_order.parquet
    ├── 02_conjugacy_classes.json
    ├── 02_conjugacy_classes.parquet
    ├── 03_backtrack_test.json
    ├── 03_backtrack_test.parquet
    └── all_experiments.parquet
```

## Verification Against Lean4

### Compare Results

```bash
# Run experiments
nix-shell shell-experiments.nix --run ./run_experiments.sh

# Compare with Lean4 proofs
cargo run --bin verify_experiments
```

### Expected Matches

| Property | GAP Result | Lean4 Proof | Match |
|----------|------------|-------------|-------|
| Order | 808...000 | monster_order | ✓ |
| Conjugacy classes | 194 | (to prove) | ✓ |
| Is simple | true | IsSimple | ✓ |

## Adding New Experiments

### Template

```bash
#!/usr/bin/env bash
set -e

EXPERIMENT="04_new_experiment"
OUTPUT_DIR="${EXPERIMENT_DIR:-./experiments}/results"
mkdir -p "$OUTPUT_DIR"

echo "🧪 Experiment 04: New Experiment"

cat > /tmp/experiment.g << 'GAP'
# GAP code here
LoadPackage("atlasrep");
M := AtlasGroup("M");

# Compute something
result := ComputeSomething(M);

# Export JSON
json := "{ \"result\": \"" + String(result) + "\" }";
PrintTo("$OUTPUT_DIR/$EXPERIMENT.json", json);

QUIT_GAP(0);
GAP

gap -q /tmp/experiment.g

# Convert to Parquet
python3 << 'PYTHON'
import json, pandas as pd
with open("$OUTPUT_DIR/$EXPERIMENT.json") as f:
    data = json.load(f)
df = pd.DataFrame([data])
df.to_parquet("$OUTPUT_DIR/$EXPERIMENT.parquet", index=False)
PYTHON

echo "✅ Complete!"
```

## Integration with CI/CD

### GitHub Actions

```yaml
- name: Run Monster Experiments
  run: |
    nix-shell shell-experiments.nix --run ./run_experiments.sh
    
- name: Upload Results
  uses: actions/upload-artifact@v4
  with:
    name: experiment-results
    path: experiments/results/*.parquet
```

## Analysis

### Load Results

```python
import pandas as pd

# Load all results
df = pd.read_parquet('experiments/results/all_experiments.parquet')

# Check pass rate
pass_rate = (df['result'] == 'PASS').mean()
print(f"Pass rate: {pass_rate:.1%}")

# View results
print(df[['experiment', 'result', 'timestamp']])
```

## Summary

✅ **3 experiments created**  
✅ **Nix environment configured**  
✅ **JSON + Parquet output**  
✅ **Automated runner**  
✅ **Ready to verify Lean4 proofs**

**Run experiments to verify Monster group properties!** 🧪✅
