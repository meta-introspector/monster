#!/usr/bin/env bash
# Experiment 03: Test BacktrackKit Conjugacy Algorithm

set -e

EXPERIMENT="03_backtrack_test"
OUTPUT_DIR="${EXPERIMENT_DIR:-./experiments}/results"
mkdir -p "$OUTPUT_DIR"

echo "🧪 Experiment 03: BacktrackKit Conjugacy Test"
echo "=============================================="
echo ""

cat > /tmp/backtrack_test.g << 'GAP'
# Load BacktrackKit
Read("gap_sage_repos/BacktrackKit/read.g");

Print("Testing BacktrackKit conjugacy algorithm...\n\n");

# Test with small symmetric group
G := SymmetricGroup(5);
Print("Test group: S5\n");
Print("Order: ", Order(G), "\n");

# Create two conjugate elements
g1 := (1,2,3);
g2 := (2,3,4);  # Conjugate to g1

Print("Element 1: ", g1, "\n");
Print("Element 2: ", g2, "\n");

# Check if conjugate
are_conjugate := g1 in ConjugacyClass(G, g2);
Print("Are conjugate: ", are_conjugate, "\n");

if are_conjugate then
    result := "PASS";
    Print("✓ VERIFIED: Elements are conjugate\n");
else
    result := "FAIL";
    Print("✗ FAILED: Not conjugate\n");
fi;

# Count conjugacy classes
num_classes := Size(ConjugacyClasses(G));
Print("Conjugacy classes in S5: ", num_classes, "\n");

json := Concatenation(
    "{\n",
    "  \"experiment\": \"03_backtrack_test\",\n",
    "  \"result\": \"", result, "\",\n",
    "  \"test_group\": \"S5\",\n",
    "  \"order\": ", String(Order(G)), ",\n",
    "  \"num_classes\": ", String(num_classes), ",\n",
    "  \"are_conjugate\": ", String(are_conjugate), "\n",
    "}\n"
);

PrintTo("$OUTPUT_DIR/$EXPERIMENT.json", json);
Print("\n✓ Results saved\n");

QUIT_GAP(0);
GAP

echo "Running GAP with BacktrackKit..."
gap -q /tmp/backtrack_test.g

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
    'test_group': data['test_group'],
    'order': data['order'],
    'num_classes': data['num_classes'],
    'are_conjugate': data['are_conjugate'],
    'timestamp': datetime.now().isoformat()
}])

df.to_parquet("$OUTPUT_DIR/$EXPERIMENT.parquet", index=False)
print(f"✓ Saved to $OUTPUT_DIR/$EXPERIMENT.parquet")
PYTHON

echo ""
echo "✅ Experiment complete!"
