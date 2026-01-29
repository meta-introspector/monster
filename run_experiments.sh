#!/usr/bin/env bash
# Run all experiments and collect results

set -e

echo "🧪 Running All Monster Group Experiments"
echo "========================================="
echo ""

export EXPERIMENT_DIR="$PWD/experiments"
mkdir -p "$EXPERIMENT_DIR/results"

# Run experiments
for exp in experiments/0*.sh; do
    if [ -f "$exp" ]; then
        echo ""
        echo "▶ Running $(basename $exp)..."
        echo "---"
        chmod +x "$exp"
        "$exp" || echo "⚠️  Experiment failed"
        echo ""
    fi
done

# Combine results
echo "📊 Combining results..."
python3 << 'PYTHON'
import pandas as pd
from pathlib import Path

results_dir = Path("experiments/results")
parquet_files = list(results_dir.glob("*.parquet"))

if parquet_files:
    dfs = [pd.read_parquet(f) for f in parquet_files]
    combined = pd.concat(dfs, ignore_index=True)
    
    combined.to_parquet("experiments/results/all_experiments.parquet", index=False)
    
    print(f"\n✓ Combined {len(dfs)} experiments")
    print(f"\nResults:")
    print(combined[['experiment', 'result']].to_string(index=False))
    
    # Summary
    passed = len(combined[combined['result'] == 'PASS'])
    total = len(combined)
    print(f"\n📊 Summary: {passed}/{total} experiments passed")
else:
    print("No results found")
PYTHON

echo ""
echo "✅ All experiments complete!"
echo "📁 Results saved to: experiments/results/"
