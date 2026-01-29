#!/usr/bin/env bash
# Experiment 01: Verify Monster Group Order

set -e

EXPERIMENT="01_monster_order"
OUTPUT_DIR="${EXPERIMENT_DIR:-./experiments}/results"
mkdir -p "$OUTPUT_DIR"

echo "🧪 Experiment 01: Monster Group Order"
echo "======================================"
echo ""

# Create GAP script
cat > /tmp/monster_order.g << 'GAP'
LoadPackage("atlasrep");
Print("Loading Monster group...\n");
M := AtlasGroup("M");
if M = fail then
    Print("ERROR: Failed to load Monster\n");
    QUIT_GAP(1);
fi;

order := Order(M);
Print("Monster order: ", order, "\n");

# Expected order
expected := 808017424794512875886459904961710757005754368000000000;
Print("Expected:      ", expected, "\n");

if order = expected then
    Print("✓ VERIFIED: Order matches!\n");
    result := "PASS";
else
    Print("✗ FAILED: Order mismatch!\n");
    result := "FAIL";
fi;

# Export JSON
json := Concatenation(
    "{\n",
    "  \"experiment\": \"01_monster_order\",\n",
    "  \"result\": \"", result, "\",\n",
    "  \"order\": \"", String(order), "\",\n",
    "  \"expected\": \"", String(expected), "\",\n",
    "  \"match\": ", String(order = expected), "\n",
    "}\n"
);

PrintTo("$OUTPUT_DIR/$EXPERIMENT.json", json);
Print("\n✓ Results saved to $OUTPUT_DIR/$EXPERIMENT.json\n");

QUIT_GAP(0);
GAP

# Run GAP
echo "Running GAP..."
gap -q /tmp/monster_order.g

# Convert to Parquet
echo ""
echo "Converting to Parquet..."
python3 << 'PYTHON'
import json
import pandas as pd
from datetime import datetime

with open("$OUTPUT_DIR/$EXPERIMENT.json") as f:
    data = json.load(f)

df = pd.DataFrame([{
    'experiment': data['experiment'],
    'result': data['result'],
    'order': data['order'],
    'expected': data['expected'],
    'match': data['match'],
    'timestamp': datetime.now().isoformat()
}])

df.to_parquet("$OUTPUT_DIR/$EXPERIMENT.parquet", index=False)
print(f"✓ Saved to $OUTPUT_DIR/$EXPERIMENT.parquet")
PYTHON

echo ""
echo "✅ Experiment complete!"
